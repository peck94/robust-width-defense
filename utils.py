import torch

from pytorch_wavelets import DWT, IDWT

class Wrapper(torch.nn.Module):
    def __init__(self, model, reconstructor):
        super().__init__()

        self.model = model
        self.reconstructor = reconstructor
    
    def forward(self, x):
        x_hat = self.reconstructor.generate(normalize(x.float()))
        return self.model(x_hat.float())

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def hard_thresh(x, lam):
    return torch.where(abs(x) < lam, torch.zeros_like(x), x)

def soft_thresh(x, lam):
    return torch.sign(x) * torch.maximum(torch.zeros_like(x), torch.abs(x) - lam)

def dwt(x, levels, method='db2'):
    xfm = DWT(J=levels, wave=method, mode='per')
    return xfm(x.float())

def idwt(coeffs, method='db2'):
    ifm = IDWT(mode='per', wave=method)
    return ifm(coeffs)
