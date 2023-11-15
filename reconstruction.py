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

    def __init__(self, eps=4/255, mu=1, iterations=10, method='fourier', **kwargs):
        """
        :param eps: Perturbation budget.
        :param mu: Parameter for soft thresholding.
        :param iterations: Number of iterations for soft thresholding.
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        """

        self.eps = eps
        self.mu = mu
        self.iterations = iterations
        
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
        trial.suggest_float('mu', 1e-3, 10, log=True)
        trial.suggest_int('iterations', 1, 100)

    def generate(self, originals):
        """
        Generate the reconstructed images.

        :param originals: Array of original inputs.
        :return: Array of reconstructed images.
        """

        # build the method if necessary
        if not self.built:
            self.method.build(self, originals)
            self.built = True
        
        # iterative soft thresholding
        coeffs = self.method.forward(normalize(originals))
        for _ in range(self.iterations):
            coeffs.soft_thresh(self.mu)

        # remove noise
        coeffs.perturb(self.eps)

        # reconstruct the sample
        x_hat = self.method.backward(coeffs)

        return x_hat
