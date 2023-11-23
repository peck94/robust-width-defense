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
        trial.suggest_float('sigma', 1e-3, 1, log=True)

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
        
        # build the sensing operator
        c = originals.shape[1]
        phi = torch.nn.Conv2d(c, c, 3, padding='same', bias=False).to(originals.device)
        psi = torch.nn.ConvTranspose2d(c, c, 3, padding=(1, 1), bias=False).to(originals.device)
        with torch.no_grad():
            torch.nn.init.normal_(phi.weight, 0, self.sigma)
            psi.weight.copy_(phi.weight)
        
        # iterative soft thresholding
        y = normalize(originals)
        coeffs = self.method.forward(torch.zeros_like(originals))
        for _ in range(self.iterations):
            coeffs = coeffs + self.method.forward(psi(y - phi(self.method.backward(coeffs).float()).float()).float())
            coeffs.soft_thresh(self.mu)

        return self.method.backward(coeffs)
