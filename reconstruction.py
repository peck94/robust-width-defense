import torch

from utils import normalize

from methods import FourierMethod, WaveletMethod, DualTreeMethod, ShearletMethod, DummyMethod

class Reconstruction:
    """
    This class instantiates the reconstruction algorithm.

    https://arxiv.org/pdf/2002.04150.pdf
    """

    def __init__(self, sigma=1, mu=1, iterations=10, method='fourier', **kwargs):
        """
        :param sigma: Variance of noise.
        :param mu: Parameter for soft thresholding.
        :param iterations: Number of iterations for soft thresholding.
        :param method: Name of the reconstruction method. See `get_method` for supported names.
        """

        self.sigma = sigma
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
        trial.suggest_float('mu', 0, 1)
        trial.suggest_int('iterations', 1, 100)
        trial.suggest_float('sigma', 0, 1)

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
        
        # generate noise
        eta = self.method.forward(self.sigma * torch.randn(originals.shape).to(originals.device))
        y = self.method.backward(self.method.forward(normalize(originals)) + eta)
        
        # iterative soft thresholding
        coeffs = self.method.forward(torch.zeros_like(originals))
        for _ in range(self.iterations):
            coeffs = coeffs + self.method.forward(y - self.method.backward(coeffs))
            coeffs.soft_thresh(self.mu)

        return self.method.backward(coeffs)
