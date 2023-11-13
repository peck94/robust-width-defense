import numpy as np

import torch

from utils import soft_thresh, hard_thresh, normalize

from subsamplers import FourierSubsampler, RandomSubsampler, DummySubsampler

from methods import FourierMethod, WaveletMethod, DualTreeMethod, ShearletMethod, DummyMethod

class Reconstruction:
    """
    This class instantiates the reconstruction algorithm.

    https://arxiv.org/pdf/2002.04150.pdf
    """

    def __init__(self, undersample_rate=0.9, lam=1, method='fourier', iterations=1, alpha=2, tol=1e-4, **kwargs):
        """
        :param undersample_rate: The undersampling rate (0 = discard everything, 1 = no undersampling).
        :param lam: Regularization parameter for reconstruction.
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        :param alpha: Parameter for adaptive thresholding.
        :param tol: Error tolerance.
        """

        self.tol = tol
        self.alpha = alpha
        self.lam = lam
        self.iterations = iterations
        self.undersample_rate = undersample_rate
        self.sampler = FourierSubsampler(self.undersample_rate, **kwargs)
        
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
        trial.suggest_float('lam', 1e-3, 10, log=True)
        trial.suggest_float('undersample_rate', .25, 1)
        trial.suggest_categorical('method', ['wavelet', 'fourier', 'dtcwt', 'shearlet'])
        trial.suggest_float('alpha', .01, 10, log=True)
        trial.suggest_categorical('iterations', list(range(1, 101)))

    def generate(self, originals):
        """
        Generate the reconstructed images.

        :param originals: Array of original inputs.
        :return: Array of reconstructed images.
        """

        # build the method if necessary
        if not self.built:
            self.method.build(self, originals)
            self.sampler.build(originals.shape)
            self.built = True
        
        # measure the signal
        x_hat = self.sampler(normalize(originals))
        y = x_hat.clone()
        
        # compute sparse representations
        coeffs = self.method.forward(x_hat)

        # ISTA: https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/08-prox-grad.pdf
        for _ in range(self.iterations):
            # update the coefficients
            coeffs = coeffs + self.method.forward(y - self.method.backward(coeffs)) * self.lam

            # threshold the coefficients
            coeffs.soft_thresh(self.alpha)

        # reconstruct the images
        x_hat = self.method.backward(coeffs)

        # clamp to [0,1]
        x_hat = torch.clamp(x_hat, 0, 1)

        return x_hat
