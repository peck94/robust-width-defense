import torch

import numpy as np

class Coefficients:
    def __init__(self):
        pass

    def get_threshold(self, lam):
        return lam

    def ht(self, x, lam):
        tau = self.get_threshold(lam)
        return torch.where(abs(x) < tau, torch.zeros_like(x), x)
    
    def st(self, x, lam):
        tau = self.get_threshold(lam)
        return torch.where(abs(x) < tau, torch.zeros_like(x), (abs(x) - tau) * x / abs(x))

    def hard_thresh(self, alpha):
        pass

    def soft_thresh(self, alpha):
        pass

    def get(self):
        pass

    def perturb(self, eps, q):
        pass

class DummyCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def get(self):
        return self.coeffs

class WaveletCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.low, self.high = coeffs

    def get_threshold(self, alpha):
        hh1 = self.high[0][:, :, -1, ...]
        n = hh1.shape[0]

        s_hat = alpha * torch.median(abs(hh1.reshape(n, -1)), dim=1).values / 0.6745
        s_y = torch.sqrt(torch.var(self.high[0], dim=(1, 2, 3, 4)))
        s_x = torch.sqrt(torch.maximum(torch.zeros_like(s_y), torch.square(s_y) - torch.square(s_hat)))

        upper = abs(self.high[0]).reshape(n, -1).max(dim=1).values
        tau = torch.minimum(torch.square(s_hat) / s_x, upper)

        return tau.reshape(n, 1, 1, 1, 1)
    
    def hard_thresh(self, alpha):
        self.high = [self.ht(x, alpha) for x in self.high]
    
    def soft_thresh(self, alpha):
        self.high = [self.st(x, alpha) for x in self.high]
    
    def get(self):
        return self.low, self.high
    
    def perturb(self, eps, q):
        masks = [2*torch.bernoulli(torch.ones(h.shape[-1]) * q).to(h.device) - 1 for h in self.high]
        self.high = [h + m*eps for h, m in zip(self.high, masks)]

class DTCWTCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.low, self.high = coeffs
    
    def hard_thresh(self, alpha):
        self.high = [self.ht(x, alpha) for x in self.high]
    
    def soft_thresh(self, alpha):
        self.high = [self.st(x, alpha) for x in self.high]
    
    def get(self):
        return self.low, self.high
    
    def perturb(self, eps, q):
        masks = [2*torch.bernoulli(torch.ones(h.shape[-1]) * q).to(h.device) - 1 for h in self.high]
        self.high = [h + m*eps for h, m in zip(self.high, masks)]

class FourierCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def hard_thresh(self, alpha):
        self.coeffs = self.ht(self.coeffs, alpha)
    
    def soft_thresh(self, alpha):
        self.coeffs = self.st(self.coeffs, alpha)
    
    def get(self):
        return self.coeffs

    def perturb(self, eps, q):
        mask = 2*torch.bernoulli(torch.ones_like(torch.real(self.coeffs)) * q) - 1
        self.coeffs = self.coeffs + eps*mask

class ShearletCoefficients(Coefficients):
    def __init__(self, coeffs, system):
        super().__init__()

        self.coeffs = coeffs
        self.system = system
    
    def get_threshold(self, alpha):
        weights = self.system.RMS * torch.ones_like(self.coeffs)
        tau = alpha * weights

        return tau

    def hard_thresh(self, alpha):
        self.coeffs = self.ht(self.coeffs, alpha)
    
    def soft_thresh(self, alpha):
        self.coeffs = self.st(self.coeffs, alpha)

    def get(self):
        return self.coeffs
    
    def perturb(self, eps, q):
        c = self.coeffs.shape[-1] - 1
        b = c // 2
        mask = 2*torch.bernoulli(torch.ones_like(self.coeffs[b:c]) * q) - 1
        self.coeffs[b:c] = self.coeffs[b:c] + eps*mask
