import numpy as np

import torch
from torch.fft import fft2, ifft2

class Subsampler:
    """
    Defines a subsampling method.

    :param undersample_rate: The undersampling rate. A value between 0 and 1 where 0 removes all information and 1 performs no subsampling.
    """

    def __init__(self, undersample_rate, **kwargs):
        self.undersample_rate = undersample_rate

    def __call__(self, originals):
        """
        Performs the subsampling operation.

        :param originals: Array of original images.
        :return: Array of subsampled images.
        """
        return originals
    
    def restore(self, y, x_hat):
        """
        Performs the inpainting operation.

        :param y: Array of subsampled images.
        :param x_hat: Array of reconstructed images.
        :return: Array of inpainted images.
        """
        return x_hat

class DummySubsampler(Subsampler):
    """
    The dummy subsampler is a no-op. It does not modify anything.
    """

    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        return originals
    
    def restore(self, y, x_hat):
        return x_hat

class RandomSubsampler(Subsampler):
    """
    The random subsampler sets a fraction of randomly selected pixels to zero.
    This fraction is equal to `1 - undersample_rate`.
    """

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
        """
        The inpainting operation restores the pixels that were not masked out.
        """
        return torch.where(self.mask > 0, y, x_hat)

class FourierSubsampler(Subsampler):
    """
    The Fourier subsampler sets a fraction of randomly selected Fourier coefficients to zero.
    """

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
        """
        The inpainting operation restores the Fourier coefficients that were not masked out.
        """
        u = fft2(y)
        v = fft2(x_hat)
        z = torch.where(self.mask > 0, u, v)
        return torch.real(ifft2(z))
