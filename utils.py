import pywt
import pytorch_wavelets
import torch
import numpy as np

def dwt(x, levels, method='bior1.3'):
    xfm = pytorch_wavelets.DWT(J=levels, wave=method, mode='symmetric')
    return xfm(x.float())

def idwt(coeffs, method='bior1.3'):
    ifm = pytorch_wavelets.IDWT(mode='symmetric', wave=method)
    return ifm(coeffs)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def soft_thresh(Yl, Yh, lam):
    def thresh(x):
        return torch.where(abs(x) < lam, torch.zeros_like(x), x)
    
    return thresh(Yl), [thresh(c) for c in Yh]

def generate_reconstructions(originals, undersample_rate=0.5, wavelet='db3', levels=3, lam=0.4, lam_decay=0.995, tol=1e-4):
    # create mask
    n = np.prod(originals.shape[2:])
    m = int(n * undersample_rate)
    mask = torch.from_numpy(np.random.permutation(
                np.concatenate(
                    (np.ones(m),
                    np.zeros(n - m))
                )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

    # undersample the signals
    y = originals * mask
    
    # reconstruct the signals
    x_hat = torch.clone(y)
    err = np.inf
    while err > tol:
        x_old = x_hat.cpu().detach().numpy()

        Yl, Yh = dwt(x_hat, levels, wavelet)
        Yl, Yh = soft_thresh(Yl, Yh, lam)
        x_hat = idwt((Yl, Yh), wavelet)

        x_hat = y + x_hat * (1 - mask)
        x_hat = torch.clamp(x_hat, 0, 1)

        lam *= lam_decay

        err = np.square(x_hat.cpu().detach().numpy() - x_old).mean()

    return x_hat
