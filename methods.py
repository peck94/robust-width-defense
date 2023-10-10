import torch
from torch.fft import fft2, ifft2

from pytorch_wavelets import DWT, IDWT, DTCWT, IDTCWT

class Method:
    """
    Defines a reconstruction method for solving the compressed sensing problem.
    """

    def __init__(self):
        pass

    def forward(self, x_hat):
        """
        The forward pass computes the sparse representation of the input according to this method.

        :param x_hat: Array of partially reconstructed images.
        :return: Sparse representation of the images.
        """
        pass

    def backward(self, z):
        """
        The backward pass computes the original images based on their sparse representations.

        :param z: Array of sparse representations.
        :return: Array of images.
        """
        pass

    @staticmethod
    def initialize_trial(trial):
        """
        Initialize the Optuna trial. Each method can have its own distinct hyperparameters.
        """
        pass

class DummyMethod(Method):
    """
    The dummy method is a no-op. It does not modify anything.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def forward(self, x_hat):
        return x_hat
    
    def backward(self, z):
        return z

class WaveletMethod(Method):
    """
    This method uses wavelets to compute the sparse representations.
    """

    def __init__(self, wavelet='db3', levels=3, **kwargs):
        """
        :param wavelet: String name of the wavelet to use.
        :param levels: Number of levels of decomposition.
        """
        super().__init__(**kwargs)

        self.wavelet = wavelet
        self.levels = levels

        self.xfm = DWT(J=self.levels, wave=self.wavelet)
        self.ifm = IDWT(wave=self.wavelet)
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_categorical('wavelet', ['sym2', 'sym8', 'sym16', 'dmey', 'db2', 'db8', 'db16'])
        trial.suggest_int('levels', 1, 10)
    
    def forward(self, x_hat):
        Xl, Xh = self.xfm(x_hat.float())
        return Xl, Xh
    
    def backward(self, z):
        return self.ifm(z)

class DualTreeMethod(Method):
    """
    This method uses the Dual Tree Complex Wavelet Transform (DTCWT) to compute the sparse representations.
    """

    def __init__(self,  levels=3, **kwargs):
        """
        :param levels: Number of levels of decomposition.
        """
        super().__init__(**kwargs)

        self.levels = levels
        self.xfm = DTCWT(J=self.levels)
        self.ifm = IDTCWT()
    
    @staticmethod
    def initialize_trial(trial):
        trial.suggest_int('levels', 1, 10)
    
    def forward(self, x_hat):
        Xl, Xh = self.xfm(x_hat.float())
        return Xl, Xh
    
    def backward(self, z):
        return self.ifm(z)

class FourierMethod(Method):
    """
    This method uses Fourier coefficients as the sparse representation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def initialize_trial(trial):
        pass
    
    def forward(self, x_hat):
        return fft2(x_hat)
    
    def backward(self, z):
        return torch.real(ifft2(z))
