import numpy as np

import torch
from torch.fft import fft2, ifft2

import pywt

from utils import dwt, idwt, hard_thresh, normalize

class Reconstruction:
    def __init__(self, tol=1e-4):
        self.tol = tol
    
    @staticmethod
    def initialize_trial(trial):
        pass

    def generate(self, originals):
        pass

class RandomSubsampling(Reconstruction):
    def __init__(self, undersample_rate=0.5, wavelet='db3', levels=3, lam=0.4, lam_decay=0.995, tol=1e-4):
        super().__init__(tol)

        self.undersample_rate = undersample_rate
        self.wavelet = wavelet
        self.levels = levels
        self.lam = lam
        self.lam_decay = lam_decay
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_categorical('wavelet', pywt.wavelist())
        trial.suggest_float('undersample_rate', 0.25, 1)
        trial.suggest_int('levels', 1, 10)
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)
    
    def generate(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        y = normalize(originals) * mask
        
        # reconstruct the signals
        x_hat = torch.clone(y)
        err = np.inf
        lam = self.lam
        while err > self.tol:
            x_old = x_hat.cpu().detach().numpy()

            Yl, Yh = dwt(x_hat, self.levels, self.wavelet)
            Yl, Yh = hard_thresh(Yl, Yh, lam)
            x_hat = idwt((Yl, Yh), self.wavelet)

            x_hat = y + x_hat * (1 - mask)
            x_hat = torch.clamp(x_hat, 0, 1)

            lam *= self.lam_decay

            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean()

        return x_hat


class FourierSubsampling(Reconstruction):
    def __init__(self, undersample_rate=0.5, wavelet='db3', levels=3, lam=0.4, lam_decay=0.995, tol=1e-4):
        super().__init__(tol)

        self.undersample_rate = undersample_rate
        self.wavelet = wavelet
        self.levels = levels
        self.lam = lam
        self.lam_decay = lam_decay
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_categorical('wavelet', pywt.wavelist())
        trial.suggest_float('undersample_rate', 0.25, 1)
        trial.suggest_int('levels', 1, 10)
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)
    
    def generate(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        y = torch.real(ifft2(fft2(normalize(originals)) * mask))
        
        # reconstruct the signals
        x_hat = torch.clone(y)
        err = np.inf
        lam = self.lam
        while err > self.tol:
            x_old = x_hat.cpu().detach().numpy()

            Yl, Yh = dwt(x_hat, self.levels, self.wavelet)
            Yl, Yh = hard_thresh(Yl, Yh, lam)
            x_hat = idwt((Yl, Yh), self.wavelet)

            x_hat = y + x_hat * (1 - mask)
            x_hat = torch.clamp(x_hat, 0, 1)

            lam *= self.lam_decay

            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean()

        return x_hat
