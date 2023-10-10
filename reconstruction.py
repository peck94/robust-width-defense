import numpy as np

import torch

from utils import hard_thresh, normalize

from subsamplers import FourierSubsampler, RandomSubsampler, DummySubsampler

from methods import FourierMethod, WaveletMethod, DualTreeMethod, DummyMethod

class Reconstruction:
    """
    This class instantiates the reconstruction algorithm. It works in two steps:
    1. Subsample the original inputs using one of the subsampling methods.
    2. Iteratively reconstruct the original inputs using one of the reconstruction methods.
    """

    def __init__(self, undersample_rate=0.5, subsample='fourier', method='fourier', lam=0.9, lam_decay=0.995, tol=1e-4, **kwargs):
        """
        :param undersample_rate: The undersampling rate for the subsampling method.
        :param subsample: Name of the subsampling method. See `get_subsampler` for supported names.
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        :param lam: Parameter for thresholding.
        :param lam_decay: Decay of the lam parameter.
        :param tol: Error tolerance.
        """

        self.undersample_rate = undersample_rate
        self.tol = tol
        self.lam = lam
        self.lam_decay = lam_decay
        
        self.subsampler = Reconstruction.get_subsampler(subsample)(undersample_rate, **kwargs)
        self.method = Reconstruction.get_method(method)(**kwargs)
    
    @staticmethod
    def get_subsampler(subsample):
        """
        Returns a subsampling method based on its name.
        """
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
        """
        Returns a reconstruction method based on its name.
        """
        if method == 'fourier':
            return FourierMethod
        elif method == 'wavelet':
            return WaveletMethod
        elif method == 'dtcwt':
            return DualTreeMethod
        elif method == 'dummy':
            return DummyMethod
        else:
            raise ValueError(f'Unsupported method: {method}')
    
    @staticmethod
    def initialize_trial(trial):
        """
        Initialize the Optuna trial.
        """
        trial.suggest_float('undersample_rate', 0.25, 1)
        trial.suggest_categorical('subsample', ['random', 'fourier'])
        trial.suggest_categorical('method', ['wavelet', 'fourier', 'dtcwt'])
        trial.suggest_float('lam', 0, 1)
        trial.suggest_float('lam_decay', 0.9, 1)

    def generate(self, originals):
        """
        Generate the reconstructed images.

        :param originals: Array of original inputs.
        :return: Array of reconstructed images.
        """

        # undersample the signals
        y = self.subsampler(normalize(originals))

        # reconstruct the signals
        err = np.inf
        lam = self.lam
        x_hat = torch.from_numpy(y.cpu().detach().numpy()).to(originals.device)
        while err > self.tol:
            # copy the images to compute the error
            x_old = x_hat.cpu().detach().numpy()

            # compute sparse representations
            z = self.method.forward(x_hat)
            # threshold the coefficients
            z = hard_thresh(z, lam)
            # reconstruct the images
            x_hat = self.method.backward(z)
            # perform the inpainting
            x_hat = self.subsampler.restore(y, x_hat)

            # clamp to [0,1]
            x_hat = torch.clamp(x_hat, 0, 1)

            # decay lam
            lam *= self.lam_decay

            # compute the error
            err = np.square(x_hat.cpu().detach().numpy() - x_old).mean() / np.square(x_old).mean()

        return x_hat
