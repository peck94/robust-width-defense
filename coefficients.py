import torch

class Coefficients:
    def __init__(self):
        pass

    def ht(self, x, lam):
        return torch.where(abs(x) < lam, torch.zeros_like(x), x)
    
    def st(self, x, lam):
        return torch.zeros_like(x) + (abs(x) - lam) / abs(x) * x * (abs(x) > lam)

    def hard_thresh(self, lam):
        pass

    def soft_thresh(self, lam):
        pass

    def get(self):
        pass

class DummyCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs
    
    def hard_thresh(self, lam):
        pass

    def soft_thresh(self, lam):
        pass

    def get(self):
        return self.coeffs

class WaveletCoefficients(Coefficients):
    def __init__(self, low, high):
        super().__init__()

        self.low = low
        self.high = high
    
    def hard_thresh(self, lam):
        self.high = [self.ht(x, lam) for x in self.high]
    
    def soft_thresh(self, lam):
        self.high = self.st(self.high, lam)
    
    def get(self):
        return self.low, self.high

class FourierCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def hard_thresh(self, lam):
        self.coeffs = self.ht(self.coeffs, lam)
    
    def soft_thresh(self, lam):
        self.coeffs = self.st(self.coeffs, lam)
    
    def get(self):
        return self.coeffs

class ShearletCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def hard_thresh(self, lam):
        self.coeffs = self.ht(self.coeffs, lam)
    
    def soft_thresh(self, lam):
        self.coeffs = self.st(self.coeffs, lam)
    
    def get(self):
        return self.coeffs
