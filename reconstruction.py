import numpy as np

import torch

from utils import normalize

from methods import FourierMethod, WaveletMethod, DualTreeMethod, ShearletMethod, DummyMethod

class Reconstruction:
    """
    This class instantiates the reconstruction algorithm.

    https://arxiv.org/pdf/2002.04150.pdf
    """

    def __init__(self, mu=1, q=.9, iterations=10, method='fourier', **kwargs):
        """
        :param mu: Parameter for soft thresholding.
        :param q: Fourier subsampling probability.
        :param iterations: Number of iterations for soft thresholding.
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        """

        self.mu = mu
        self.q = q
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
        trial.suggest_float('mu', 0, 1)
        trial.suggest_float('q', .5, 1)
        trial.suggest_int('iterations', 1, 100)

    def generate(self, originals, return_phi=False):
        """
        Generate the reconstructed images.

        :param originals: Array of original inputs.
        :param return_phi: If True, returns the (phi, psi) tuple as well.
        :return: Array of reconstructed images.
        """

        # build the method if necessary
        if not self.built:
            self.method.build(self, originals)
            self.built = True
        
        # build the sensing operator
        mask = torch.bernoulli(torch.ones_like(originals) * self.q)
        phi = lambda x: torch.fft.fft2(x) * mask
        psi = lambda x: torch.real(torch.fft.ifft2(x)).float()
        
        # iterative soft thresholding
        y = phi(normalize(originals))
        coeffs = self.method.forward(torch.zeros_like(originals))
        for _ in range(self.iterations):
            coeffs = coeffs + self.method.forward(psi(y - phi(self.method.backward(coeffs))))
            coeffs.soft_thresh(self.mu)

            coeffs = self.method.forward(torch.clamp(self.method.backward(coeffs), 0, 1))

        if return_phi:
            return normalize(self.method.backward(coeffs)), phi, psi
        else:
            return normalize(self.method.backward(coeffs))

    def certify(self, originals, iterations=10):
        norms = []
        for _ in range(iterations):
            y2, phi, psi = self.generate(originals, return_phi=True)
            y1 = phi(normalize(originals))

            c1 = self.method.forward(psi(y1))
            c2 = self.method.forward(psi(y2))

            d = abs(c1 - c2).sum().cpu().detach().numpy()
            norms.append(d)
        return np.mean(norms)
