import torch

class Coefficients:
    def __init__(self):
        pass

    def ht(self, x, lam):
        tau = self.get_threshold(lam)
        return torch.where(abs(x) < tau, torch.zeros_like(x), x)
    
    def st(self, x, lam):
        tau = self.get_threshold(lam)
        return torch.zeros_like(x) + (abs(x) - tau) / abs(x) * x * (abs(x) > tau)

    def hard_thresh(self, alpha):
        pass

    def soft_thresh(self, alpha):
        pass

    def get_threshold(self, alpha):
        return alpha

    def get(self):
        pass

class DummyCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs
    
    def hard_thresh(self, alpha):
        pass

    def soft_thresh(self, alpha):
        pass

    def get(self):
        return self.coeffs

class WaveletCoefficients(Coefficients):
    def __init__(self, low, high):
        super().__init__()

        self.low = low
        self.high = high
    
    def hard_thresh(self, alpha):
        self.high = [self.ht(x, alpha) for x in self.high]
    
    def soft_thresh(self, alpha):
        self.high = [self.st(x, alpha) for x in self.high]
    
    def get(self):
        return self.low, self.high

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

class ShearletCoefficients(Coefficients):
    def __init__(self, coeffs, method):
        super().__init__()

        self.coeffs = coeffs
        self.method = method
    
    def get_threshold(self, alpha):
        weights = self.method.system.RMS * torch.ones_like(self.coeffs)
        return alpha * weights * self.method.sigma

    def hard_thresh(self, alpha):
        self.coeffs = self.ht(self.coeffs, alpha)
    
    def soft_thresh(self, alpha):
        self.coeffs = self.st(self.coeffs, alpha)

    def get(self):
        return self.coeffs
