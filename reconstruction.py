import numpy as np

import torch

from utils import soft_thresh, hard_thresh, normalize

from subsamplers import FourierSubsampler, RandomSubsampler, DummySubsampler

from methods import FourierMethod, WaveletMethod, DualTreeMethod, ShearletMethod, DummyMethod

class Reconstruction:
    """
    This class instantiates the reconstruction algorithm. It works in two steps:
    1. Subsample the original inputs using one of the subsampling methods.
    2. Iteratively reconstruct the original inputs using one of the reconstruction methods.
    """

    def __init__(self, method='fourier', lam=0.9, sigma=0.1, tol=1e-4, **kwargs):
        """
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        :param lam: Parameter for thresholding.
        :param sigma: Standard deviation of the noise.
        :param tol: Error tolerance.
        """

        self.tol = tol
        self.lam = lam
        self.sigma = sigma
        
        self.method = Reconstruction.get_method(method)(**kwargs)
        self.built = False

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
        elif method == 'shearlet':
            return ShearletMethod
        else:
            raise ValueError(f'Unsupported method: {method}')
    
    @staticmethod
    def initialize_trial(trial):
        """
        Initialize the Optuna trial.
        """
        trial.suggest_categorical('method', ['wavelet', 'fourier', 'dtcwt', 'shearlet'])
        trial.suggest_float('lam', .01, 1e4, log=True)
        trial.suggest_float('sigma', 0.001, 1, log=True)

    def generate(self, originals):
        """
        Generate the reconstructed images.

        :param originals: Array of original inputs.
        :return: Array of reconstructed images.
        """

        # build the method if necessary
        if not self.built:
            self.method.build(originals)
            self.built = True

        # compute noise
        noise = self.sigma * torch.randn(originals.shape).to(originals.device)

        # compute sparse representations
        z = self.method.forward(normalize(originals) + noise)

        # threshold the coefficients
        z = hard_thresh(z, self.lam)

        # reconstruct the images
        x_hat = self.method.backward(z)

        # clamp to [0,1]
        x_hat = torch.clamp(x_hat, 0, 1)

        return x_hat
