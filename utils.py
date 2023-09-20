import torch
import pytorch_wavelets

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

def hard_thresh(Yl, Yh, lam):
    def thresh(x):
        return torch.where(abs(x) < lam, torch.zeros_like(x), x)
    
    return thresh(Yl), [thresh(c) for c in Yh]

def dwt(x, levels, method='bior1.3'):
    xfm = pytorch_wavelets.DWT(J=levels, wave=method, mode='symmetric')
    return xfm(x.float())

def idwt(coeffs, method='bior1.3'):
    ifm = pytorch_wavelets.IDWT(mode='symmetric', wave=method)
    return ifm(coeffs)
