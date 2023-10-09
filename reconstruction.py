import numpy as np

import torch
from torch.fft import fft2, ifft2

from utils import dwt, idwt, hard_thresh, normalize

class Subsampler:
    def __init__(self, undersample_rate, **kwargs):
        self.undersample_rate = undersample_rate

    def __call__(self, originals):
        return originals
    
    def restore(self, y, x_hat):
        return x_hat

class DummySubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        return originals
    
    def restore(self, y, x_hat):
        return x_hat

class RandomSubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        self.mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        return originals * self.mask
    
    def restore(self, y, x_hat):
        return torch.where(self.mask > 0, y, x_hat)

class FourierSubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        self.mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        return torch.real(ifft2(fft2(originals) * self.mask))
    
    def restore(self, y, x_hat):
        u = fft2(y)
        v = fft2(x_hat)
        z = torch.where(self.mask > 0, u, v)
        return torch.real(ifft2(z))

class Reconstruction:
    def __init__(self, undersample_rate=0.5, subsample='fourier', method='fourier', lam=0.9, lam_decay=0.995, tol=1e-4, **kwargs):
        self.undersample_rate = undersample_rate
        self.tol = tol
        self.lam = lam
        self.lam_decay = lam_decay
        
        self.subsampler = Reconstruction.get_subsampler(subsample)(undersample_rate, **kwargs)
        self.method = Reconstruction.get_method(method)(**kwargs)
    
    @staticmethod
    def get_subsampler(subsample):
        if subsample == 'fourier':
            return FourierSubsampler
        elif subsample == 'random':
            return RandomSubsampler
        elif subsample == 'dummy':
            return DummySubsampler
        else:
            raise ValueError(f'Unsupported subsampler: {subsample}')

    @staticmethod
    def get_method(method):
        if method == 'fourier':
            return FourierMethod
        elif method == 'wavelet':
            return WaveletMethod
        elif method == 'dummy':
            return DummyMethod
        else:
            raise ValueError(f'Unsupported method: {method}')
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_float('undersample_rate', 0.25, 1)
        trial.suggest_categorical('subsample', ['random', 'fourier'])
        trial.suggest_categorical('method', ['wavelet', 'fourier'])
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)

    def generate(self, originals):
        # undersample the signals
        y = self.subsampler(normalize(originals))

        # reconstruct the signals
        err = np.inf
        lam = self.lam
        x_hat = torch.from_numpy(y.cpu().detach().numpy())
        while err > self.tol:
            x_old = x_hat.cpu().detach().numpy()

            z = self.method.forward(x_hat)
            z = hard_thresh(z, lam)
            x_hat = self.method.backward(z)
            x_hat = self.subsampler.restore(y, x_hat)

            x_hat = torch.clamp(x_hat, 0, 1)
            lam *= self.lam_decay

            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean() / np.square(x_old).mean()

        return x_hat

class Method:
    def __init__(self):
        pass

    def forward(self, x_hat):
        pass

    def backward(self, z):
        pass

    @staticmethod
    def initialize_trial(trial):
        pass

class WaveletMethod(Method):
    def __init__(self, wavelet='db3', levels=3, **kwargs):
        super().__init__(**kwargs)

        self.wavelet = wavelet
        self.levels = levels
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_categorical('wavelet', ['sym2', 'sym8', 'sym16', 'dmey', 'db2', 'db8', 'db16', 'dtcwt'])
        trial.suggest_int('levels', 1, 10)
    
    def forward(self, x_hat):
        Xl, Xh = dwt(x_hat, self.levels, self.wavelet)
        return Xl, Xh
    
    def backward(self, z):
        return idwt(z, self.wavelet)

class FourierMethod(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def forward(self, x_hat):
        return fft2(x_hat)
    
    def backward(self, z):
        return torch.real(ifft2(z))

class DummyMethod(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def forward(self, x_hat):
        return x_hat
    
    def backward(self, z):
        return z
