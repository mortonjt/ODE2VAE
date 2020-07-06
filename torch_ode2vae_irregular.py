import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils import data
from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl
from torch_bnn import BNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchdiffeq import odeint

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from multiprocessing import Process, freeze_support
torch.multiprocessing.set_start_method('spawn', force="True")


def obscure_func(X):
    N, T, D = X.shape
    mask = np.random.rand(N, T) < 0.1
    lengths = np.minimum(np.random.poisson(T, size=N), T - 1)

    ts = np.vstack([np.arange(T) for _ in range(N)])
    ts[mask] = -1
    for i in range(N):
        ts[i, lengths[i]:] = -1

    # keep all initial time points
    ts[:, 0] = 0

    # filter out masked timepoints
    times = []
    for i in range(N):
        t = ts[i]
        t = list(t[t > -1]) + [-1] * np.sum(t==-1)
        times.append(t)

    # filter out masked images and pad
    trainX = []
    for i in range(N):
        t = np.array(times[i])
        idx = (t >= 0)
        x_ = X[i, idx]
        if np.sum(t >= 0) > 0:
            ones = -1 * np.ones((np.sum(t < 0), D))
            x_ = np.vstack((x_, ones))
        trainX.append(x_)

    trainX = np.stack(trainX, 0)
    times = np.stack(times, 0)
    return trainX, times, lengths

def default_func(X):
    N, T, D = X.shape
    ts = np.vstack([np.arange(T) for _ in range(N)])
    lengths = np.array([T] * N)
    return X, ts, lengths

# prepare dataset
class Dataset(data.Dataset):
    def __init__(self, Xtr, ts, lengths):
        self.Xtr = Xtr           # N,16,784
        self.ts = ts             # N,16
        self.lengths = lengths   # N
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx], self.ts[idx], self.lengths[idx]

# read data
X = loadmat('rot-mnist-3s.mat')['X'].squeeze() # (N, 16, 784)
N = 500
T = 16
train_X, train_ts, train_lengths = obscure_func(X[:N])
test_X, test_ts, test_lengths = default_func(X[N:])
Xtr   = torch.tensor(train_X,dtype=torch.float32).view([N,T,1,28,28])
Xtest = torch.tensor(test_X,dtype=torch.float32).view([-1,T,1,28,28])
# Generators
params = {'batch_size': 100, 'shuffle': True, 'num_workers': 4}
trainset = Dataset(Xtr, torch.tensor(train_ts), torch.tensor(train_lengths))
trainset = data.DataLoader(trainset, **params)
testset  = Dataset(Xtest, torch.tensor(test_ts),
                   torch.tensor(test_lengths))
testset  = data.DataLoader(testset, **params)

# utils
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.w = w
    def forward(self, input):
        nc = input[0].numel()//(self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)

def mask(X, lengths):
    for n, l in enumerate(lengths):
        X[:, n, int(l):] = 0
    return X


def pad(x, T):
    p = T - len(x)
    shape = [p] + list(x.shape[1:])
    padding = torch.zeros(shape, device=x.device)
    res = torch.cat([x, padding], 0)
    return res

def get_time_indices_single(grid, query):
    """ Retrieves times specified by query.

    Parameters
    ----------
    grid : torch.Tensor
        Equally spaced time points separated by dt.
        Dimension T.
    query : torch.Tensor
        Irregular spaced time points being queried.
        Dimension t.

    Returns
    -------
    index : torch.Tensor
        Locations on the time_grid where the query times are.

    Notes
    -----
    There are a few assumptions being made here
    1. Time is already normalized to begin at zero.
    2. The query times are already in the grid,
       and that dt spacing can completely resolve them.

    This implementation came from here
    https://discuss.pytorch.org/t/find-the-nearest-value-in-the-list/73772/3
    """
    norm_query = query.view(-1, 1)
    comps = grid > norm_query
    _, index = torch.min(comps, dim=-1)
    return index + 1


def get_time_indices_batch(grid, query):
    B, _ = query.size()
    res = []
    for b in range(B):
        idx = get_time_indices_single(grid, query[b])
        res.append(idx)

    return torch.stack(res)

def dense_integrate(f, z0, ts, dt, method, ret_time_grid=False):
    device = ts.device
    input_tuple = isinstance(z0, tuple)
    T = torch.max(ts)    # T
    # dense integration grid
    td = torch.arange(0, T, dt, dtype=torch.float32, device=device)
    ts_idx = get_time_indices_batch(td, ts)
    N_, T_ = ts.shape

    zd = odeint(f, z0, td, method=method)  # T,N,n # dense sequence
    if not input_tuple:
        shape = list(zd.shape)
        shape[0] = T_
        z_ = torch.zeros(shape, device=device)
        for b in range(N_):
            idx = ts_idx[b]
            z[:, b] = zd[idx, b]
    else:
        z = []
        for zd_ in zd:
            shape = list(zd_.shape)
            shape[0] = T_
            z_ = torch.zeros(shape, device=device)
            for b in range(N_):
                idx = ts_idx[b]
                z_[:, b] = zd_[idx, b]
            z.append(z_)
    if ret_time_grid:
        return z, zd, td
    return z, zd


# model implementation
class ODE2VAE(nn.Module):
    def __init__(self, n_filt=8, q=8):
        super(ODE2VAE, self).__init__()
        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, 2*q)
        self.fc2 = nn.Linear(h_dim, 2*q)
        self.fc3 = nn.Linear(q, h_dim)
        # differential function
        # to use a deterministic differential function, set bnn=False and self.beta=0.0
        self.bnn = BNN(2*q, q, n_hid_layers=2, n_hidden=50, act='celu', layer_norm=True, bnn=True)
        # downweighting the BNN KL term is helpful if self.bnn is heavily overparameterized
        self.beta = 1.0 # 2*q/self.bnn.kl().numel()
        # decoder
        self.decoder = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )
        self._zero_mean = torch.zeros(2*q).to(device)
        self._eye_covar = torch.eye(2*q).to(device)
        self.mvn = MultivariateNormal(self._zero_mean, self._eye_covar)

    def ode2vae_rhs(self,t,vs_logp,f):
        vs, logp = vs_logp # N,2q & N
        q = vs.shape[1]//2
        dv = f(vs) # N,q
        ds = vs[:,:q]  # N,q
        dvs = torch.cat([dv,ds],1) # N,2q
        ddvi_dvi = torch.stack(
                    [torch.autograd.grad(dv[:,i],vs,torch.ones_like(dv[:,i]),
                    retain_graph=True,create_graph=True)[0].contiguous()[:,i]
                    for i in range(q)],1) # N,q --> df(x)_i/dx_i, i=1..q
        tr_ddvi_dvi = torch.sum(ddvi_dvi,1) # N
        return (dvs,-tr_ddvi_dvi)

    def elbo(self, qz_m, qz_logv, zode_L, logpL, X, XrecL, lengths,
             Ndata, qz_enc_m=None, qz_enc_logv=None):
        ''' Input:
                qz_m        - latent means [N,2q]
                qz_logv     - latent logvars [N,2q]
                zode_L      - latent trajectory samples [L,N,T,2q]
                logpL       - densities of latent trajectory samples [L,N,T]
                X           - input images [N,T,nc,d,d]
                XrecL       - reconstructions [L,N,T,nc,d,d]
                lengths     - time series lengths [N]
                Ndata       - number of sequences in the dataset (required for elbo
                qz_enc_m    - encoder density means  [N*T,2*q]
                qz_enc_logv - encoder density variances [N*T,2*q]
            Returns:
                likelihood
                prior on ODE trajectories KL[q_ode(z_{0:T})||N(0,I)]
                prior on BNN weights
                instant encoding term KL[q_ode(z_{0:T})||q_enc(z_{0:T}|X_{0:T})] (if required)
        '''
        [N,T,nc,d,d] = X.shape
        L = zode_L.shape[0]
        q = qz_m.shape[1]//2
        # prior
        log_pzt = self.mvn.log_prob(zode_L.contiguous().view([L*N*T,2*q])) # L*N*T
        log_pzt = log_pzt.view([L,N,T]) # L,N,T
        kl_zt   = logpL - log_pzt  # L,N,T
        kl_z    = kl_zt.sum(2).mean(0) # N
        kl_w    = self.bnn.kl().sum()
        eps = 1e-3
        # likelihood
        XL = X.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d
        lhood_L = torch.log(eps+XrecL)*XL + torch.log(eps+1-XrecL)*(1-XL) # L,N,T,nc,d,d
        lhood_L = mask(lhood_L, lengths)
        lhood = lhood_L.sum([2,3,4,5]).mean(0) # N
        if qz_enc_m is not None: # instant encoding
            qz_enc_mL    = qz_enc_m.repeat([L,1])  # L*N*T,2*q
            qz_enc_logvL = qz_enc_logv.repeat([L,1])  # L*N*T,2*q
            mean_ = qz_enc_mL.contiguous().view(-1) # L*N*T*2*q
            std_  = eps+qz_enc_logvL.exp().contiguous().view(-1) # L*N*T*2*q
            qenc_zt_ode = Normal(mean_,std_).log_prob(
                zode_L.contiguous().view(-1)).view([L,N,T,2*q])
            qenc_zt_ode = mask(qenc_zt_ode, lengths)
            qenc_zt_ode = qenc_zt_ode.sum([3]) # L,N,T
            inst_enc_KL = logpL - qenc_zt_ode
            inst_enc_KL = inst_enc_KL.sum(2).mean(0) # N
            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_w, Ndata*inst_enc_KL.mean()
        else:
            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_w

    def forward(self, X, ts, lengths, Ndata, L=1, inst_enc=False, method='dopri5', dt=0.1):
        ''' Input
                X          - input images [N,T,nc,d,d]
                ts         - observed time points
                lengths    - length of observed time series
                Ndata      - number of sequences in the dataset (required for elbo)
                L          - number of Monta Carlo draws (from BNN)
                inst_enc   - whether instant encoding is used or not
                method     - numerical integration method
                dt         - numerical integration step size
            Returns
                Xrec_mu    - reconstructions from the mean embedding - [N,nc,D,D]
                Xrec_L     - reconstructions from latent samples     - [L,N,nc,D,D]
                qz_m       - mean of the latent embeddings           - [N,q]
                qz_logv    - log variance of the latent embeddings   - [N,q]
                lhood-kl_z - ELBO
                lhood      - reconstruction likelihood
                kl_z       - KL
        '''
        # encode
        [N,T,nc,d,d] = X.shape
        h = self.encoder(X[:,0])
        qz0_m, qz0_logv = self.fc1(h), self.fc2(h) # N,2q & N,2q
        q = qz0_m.shape[1]//2
        # latent samples
        eps   = torch.randn_like(qz0_m)  # N,2q
        z0    = qz0_m + eps*torch.exp(qz0_logv) # N,2q
        logp0 = self.mvn.log_prob(eps) # N
        # ODE
        ztL   = []
        logpL = []
        # sample L trajectories
        for l in range(L):
            f       = self.bnn.draw_f() # draw a differential function
            oderhs  = lambda t,vs: self.ode2vae_rhs(t,vs,f) # make the ODE forward function
            # zt,logp = odeint(oderhs,(z0,logp0),t,method=method) # T,N,2q & T,N
            z, zd = dense_integrate(oderhs, (z0, logp0), ts, dt, method=method)
            zt, logp = z
            zt, logp = pad(zt, T), pad(logp, T)
            ztL.append(zt.permute([1,0,2]).unsqueeze(0)) # 1,N,T,2q
            logpL.append(logp.permute([1,0]).unsqueeze(0)) # 1,N,T
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        logpL = torch.cat(logpL) # L,N,T
        # decode
        st_muL = ztL[:,:,:,q:] # L,N,T,q
        s = self.fc3(st_muL.contiguous().view([L*N*T,q]) ) # L*N*T,h_dim
        Xrec = self.decoder(s) # L*N*T,nc,d,d
        Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d
        # likelihood and elbo
        if inst_enc:
            h = self.encoder(X.contiguous().view([N*T,nc,d,d]))
            qz_enc_m, qz_enc_logv = self.fc1(h), self.fc2(h) # N*T,2q & N*T,2q
            lhood, kl_z, kl_w, inst_KL = \
                self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, lengths,
                          Ndata, qz_enc_m, qz_enc_logv)
            elbo = lhood - kl_z - inst_KL - self.beta*kl_w
        else:
            lhood, kl_z, kl_w = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec,
                                          lengths, Ndata)
            elbo = lhood - kl_z - self.beta*kl_w
        return Xrec, qz0_m, qz0_logv, ztL, elbo, lhood, kl_z, self.beta*kl_w

    def mean_rec(self, X, method='dopri5', dt=0.1):
        [N,T,nc,d,d] = X.shape
        # encode
        h = self.encoder(X[:,0])
        qz0_m = self.fc1(h) # N,2q
        q = qz0_m.shape[1]//2
        # ode
        def ode2vae_mean_rhs(t,vs,f):
            q = vs.shape[1]//2
            dv = f(vs) # N,q
            ds = vs[:,:q]  # N,q
            return torch.cat([dv,ds],1) # N,2q
        f     = self.bnn.draw_f(mean=True) # use the mean differential function
        odef  = lambda t,vs: ode2vae_mean_rhs(t,vs,f) # make the ODE forward function
        t     = dt * torch.arange(T,dtype=torch.float).to(qz0_m.device)
        zt_mu = odeint(odef,qz0_m,t,method=method).permute([1,0,2]) # N,T,2q
        # decode
        st_mu = zt_mu[:,:,q:] # N,T,q
        s = self.fc3(st_mu.contiguous().view([N*T,q]) ) # N*T,q
        Xrec_mu = self.decoder(s) # N*T,nc,d,d
        Xrec_mu = Xrec_mu.view([N,T,nc,d,d]) # N,T,nc,d,d
        # error
        mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu,mse

# plotting
def plot_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png'):
    N = min(X.shape[0],10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T = X.shape[1]
    plt.figure(2,(T,3*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+T+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()


if __name__ == '__main__':
    freeze_support()
    ode2vae = ODE2VAE(q=8,n_filt=16).to(device)
    Nepoch = 500
    optimizer = torch.optim.Adam(ode2vae.parameters(),lr=1e-3)
    for ep in range(Nepoch):
        L = 1 if ep<Nepoch//2 else 5 # increasing L as optimization proceeds is a good practice
        for i,local_batch in enumerate(trainset):
            trainX, trainTS, trainL = local_batch
            trainX  = trainX.to(device)
            trainTS  = trainTS.to(device)
            trainL  = trainL.to(device)
            elbo, lhood, kl_z, kl_w = ode2vae(
                trainX, trainTS, trainL, len(trainset), L=L, inst_enc=True, method='rk4')[4:]
            tr_loss = -elbo
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f}  kl_w:{:8.2f}'.\
                format(i, lhood.item(), kl_z.item(), kl_w.item()))
        with torch.set_grad_enabled(False):
            for test_batch in testset:
                test_batch = test_batch[0].to(device)
                Xrec_mu, test_mse = ode2vae.mean_rec(test_batch, method='rk4')
                plot_rot_mnist(test_batch, Xrec_mu, False, fname='rot_mnist_irregular.png')
                torch.save(ode2vae.state_dict(), 'ode2vae_mnist.pth')
                break
        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, Nepoch, tr_loss.item(), test_mse.item()))
