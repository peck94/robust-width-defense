import numpy as np

import torch
from torch.fft import fft2, ifft2

class Subsampler:
    """
    Defines a subsampling method.

    :param undersample_rate: The undersampling rate. A value between 0 and 1 where 0 removes all information and 1 performs no subsampling.
    """

    def __init__(self, undersample_rate, device, **kwargs):
        self.undersample_rate = undersample_rate
        self.device = device

    def __call__(self, originals):
        """
        Performs the subsampling operation.

        :param originals: Array of original images.
        :return: Array of subsampled images.
        """
        return originals

class DummySubsampler(Subsampler):
    """
    The dummy subsampler is a no-op. It does not modify anything.
    """

    def __init__(self, undersample_rate, **kwargs):
        super().__init__(undersample_rate, **kwargs)
    
    def __call__(self, originals):
        return originals

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
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(self.device)
        
        return originals * mask

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
        mask = torch.from_numpy(np.random.permutation(
                    np.concatenate(
                        (np.ones(m),
                        np.zeros(n - m))
                    )).reshape(1, 1, *originals.shape[2:])).to(self.device)
        
        return torch.real(ifft2(fft2(originals) * mask))
