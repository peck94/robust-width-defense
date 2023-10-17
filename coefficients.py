import torch

class Coefficients:
    def __init__(self):
        pass

    def hard_thresh(self, lam):
        pass

    def soft_thresh(self, lam):
        pass

class WaveletCoefficients(Coefficients):
    def __init__(self, low, high):
        super().__init__()

        self.low = low
        self.high = high
    
    def hard_thresh(self, lam):
        self.high = torch.where(abs(self.high) < lam, torch.zeros_like(self.high), self.high)
    
    def soft_thresh(self, lam):
        self.high = torch.zeros_like(self.high) + (abs(self.high) - lam) / abs(self.high) * self.high * (abs(self.high) > lam)

class FourierCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def hard_thresh(self, lam):
        self.coeffs = torch.where(abs(self.coeffs) < lam, torch.zeros_like(self.coeffs), self.coeffs)
    
    def soft_thresh(self, lam):
        self.coeffs = torch.zeros_like(self.coeffs) + (abs(self.coeffs) - lam) / abs(self.coeffs) * self.coeffs * (abs(self.coeffs) > lam)

class ShearletCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def hard_thresh(self, lam):
        self.coeffs = torch.where(abs(self.coeffs) < lam, torch.zeros_like(self.coeffs), self.coeffs)
    
    def soft_thresh(self, lam):
        self.coeffs = torch.zeros_like(self.coeffs) + (abs(self.coeffs) - lam) / abs(self.coeffs) * self.coeffs * (abs(self.coeffs) > lam)
