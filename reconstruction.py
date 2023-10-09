import numpy as np

import torch
from torch.fft import fft2, ifft2

from utils import dwt, idwt, hard_thresh, normalize

class Subsampler:
    def __init__(self, undersample_rate, **kwargs):
        self.undersample_rate = undersample_rate

    def __call__(self, originals):
        return originals

class DummySubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        return originals

class RandomSubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        return originals * mask

class FourierSubsampler(Subsampler):
    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        # create mask
        n = np.prod(originals.shape[2:])
        m = int(n * self.undersample_rate)
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(originals.device)

        # undersample the signals
        return torch.real(ifft2(fft2(originals) * mask))

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
        self.method.initialize(y)

        # reconstruct the signals
        err = np.inf
        lam = self.lam
        while err > self.tol:
            x_old = y.cpu().detach().numpy()

            y = self.method.reconstruct(y, lam)
            y = torch.clamp(y, 0, 1)

            lam *= self.lam_decay

            err = np.square(y.cpu().detach().numpy() - x_old).mean()

        return y

class Method:
    def __init__(self):
        pass

    def initialize(self, y):
        pass

    def reconstruct(self, x_hat, lam):
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
        trial.suggest_categorical('wavelet', ['sym2', 'sym8', 'sym16', 'dmey', 'db2', 'db8', 'db16'])
        trial.suggest_int('levels', 1, 10)
    
    def initialize(self, y):
        self.Yl, self.Yh = dwt(y, self.levels, self.wavelet)
    
    def reconstruct(self, x_hat, lam):
        Xl, Xh = dwt(x_hat, self.levels, self.wavelet)
        Xl = hard_thresh(Xl, lam)
        Xh = [hard_thresh(xh, lam) for xh in Xh]

        return idwt((Xl, Xh), self.wavelet)

class FourierMethod(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def initialize(self, y):
        self.y = y
    
    def reconstruct(self, x_hat, lam):
        z = hard_thresh(fft2(x_hat), lam)
        return torch.real(ifft2(z))

class DummyMethod(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def initialize(self, y):
        pass
    
    def reconstruct(self, x_hat, lam):
        return x_hat
