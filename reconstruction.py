import numpy as np

import torch
from torch.fft import fft2, ifft2

import pywt

from utils import dwt, idwt, hard_thresh, normalize

class Reconstruction:
    def __init__(self, undersample_rate=0.5, subsample='fourier', tol=1e-4):
        self.undersample_rate = undersample_rate
        self.tol = tol
        self.subsample_type = subsample
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_float('undersample_rate', 0.25, 1)
        trial.suggest_categorical('subsample', ['random', 'fourier'])

    def generate(self, originals):
        pass

    def subsample(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        if self.subsample_type == 'random':
            return normalize(originals) * mask
        elif self.subsample_type == 'fourier':
            return torch.real(fft2(normalize(originals)) * mask)
        else:
            raise ValueError(f'Unsupported subsampling method: {self.subsample}')

class RandomSubsampling(Reconstruction):
    def __init__(self, wavelet='db3', levels=3, lam=0.4, lam_decay=0.995, **kwargs):
        super().__init__(**kwargs)

        self.wavelet = wavelet
        self.levels = levels
        self.lam = lam
        self.lam_decay = lam_decay
    
    @staticmethod
    def initialize_trial(trial):
        Reconstruction.initialize_trial(trial)
        trial.suggest_categorical('wavelet', pywt.wavelist())
        trial.suggest_int('levels', 1, 10)
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)
    
    def generate(self, originals):
        # undersample the signals
        y = self.subsample(originals)
        Yl, Yh = dwt(y, self.levels, self.wavelet)

        # reconstruct the signals
        x_hat = torch.zeros_like(y)
        err = np.inf
        lam = self.lam
        while err > self.tol:
            x_old = x_hat.cpu().detach().numpy()

            Xl, Xh = dwt(x_hat, self.levels, self.wavelet)
            Zl = Yl - Xl
            Zh = [yh - xh for yh, xh in zip(Yh, Xh)]

            z = idwt((Zl, Zh), self.wavelet)
            x_hat = hard_thresh(x_hat + z, lam)
            x_hat = torch.clamp(x_hat, 0, 1)

            lam *= self.lam_decay

            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean()

        return x_hat

class FourierSubsampling(Reconstruction):
    def __init__(self, wavelet='db3', lam=0.4, lam_decay=0.995, **kwargs):
        super().__init__(**kwargs)

        self.wavelet = wavelet
        self.lam = lam
        self.lam_decay = lam_decay
    
    @staticmethod
    def initialize_trial(trial):
        Reconstruction.initialize_trial(trial)
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)
    
    def generate(self, originals):
        # undersample the signals
        y = self.subsample(originals)
        
        # reconstruct the signals
        x_hat = torch.zeros_like(y)
        err = np.inf
        lam = self.lam
        while err > self.tol:
            x_old = x_hat.cpu().detach().numpy()

            x_hat = hard_thresh(x_hat + torch.real(ifft2(y - fft2(x_hat))), lam)
            x_hat = torch.clamp(x_hat, 0, 1)

            lam *= self.lam_decay

            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean()

        return x_hat
