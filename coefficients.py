import torch

class Coefficients:
    def __init__(self):
        pass

    def clone(self):
        return Coefficients(self.get())

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

    def __add__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __sub__(self, other):
        return self + (-1) * other
    
    def __abs__(self):
        pass

    def sum(self):
        pass
    
    def norm(self):
        return abs(self).sum()

class DummyCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs

    def get(self):
        return self.coeffs
    
    def clone(self):
        return DummyCoefficients(self.get())

    def __add__(self, other):
        return self
    
    def __rmul__(self, other):
        return self
    
    def __abs__(self):
        return self

    def sum(self):
        return 0

class WaveletCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.low, self.high = coeffs
    
    def clone(self):
        return WaveletCoefficients(self.get())
    
    def hard_thresh(self, alpha):
        self.high = [self.ht(x, alpha) for x in self.high]
    
    def soft_thresh(self, alpha):
        self.high = [self.st(x, alpha) for x in self.high]
    
    def get(self):
        return self.low, self.high

    def __add__(self, other):
        low = self.low + other.low
        high = [h1 + h2 for h1, h2 in zip(self.high, other.high)]
        return WaveletCoefficients((low, high))
    
    def __rmul__(self, other):
        low = self.low * other
        high = [other * h for h in self.high]
        return WaveletCoefficients((low, high))
    
    def __abs__(self):
        low = abs(self.low)
        high = [abs(h) for h in self.high]
        return WaveletCoefficients((low, high))

    def sum(self):
        hs = torch.cat([h.reshape(h.shape[0], -1) for h in self.high], 1)
        return torch.sum(self.low.reshape(self.low.shape[0], -1), dim=1) + torch.sum(hs, dim=1)

class DTCWTCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.low, self.high = coeffs
    
    def clone(self):
        return DTCWTCoefficients(self.get())
    
    def hard_thresh(self, alpha):
        self.high = [self.ht(x, alpha) for x in self.high]
    
    def soft_thresh(self, alpha):
        self.high = [self.st(x, alpha) for x in self.high]
    
    def get(self):
        return self.low, self.high

    def __add__(self, other):
        low = self.low + other.low
        high = [h1 + h2 for h1, h2 in zip(self.high, other.high)]
        return DTCWTCoefficients((low, high))
    
    def __rmul__(self, other):
        low = self.low * other
        high = [other * h for h in self.high]
        return DTCWTCoefficients((low, high))
    
    def __abs__(self):
        low = abs(self.low)
        high = [abs(h) for h in self.high]
        return DTCWTCoefficients((low, high))

    def sum(self):
        hs = torch.cat([h.reshape(h.shape[0], -1) for h in self.high], 1)
        return torch.sum(self.low.reshape(self.low.shape[0], -1), dim=1) + torch.sum(hs, dim=1)

class FourierCoefficients(Coefficients):
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = coeffs
    
    def clone(self):
        return FourierCoefficients(self.get())

    def hard_thresh(self, alpha):
        self.coeffs = self.ht(self.coeffs, alpha)
    
    def soft_thresh(self, alpha):
        self.coeffs = self.st(self.coeffs, alpha)
    
    def get(self):
        return self.coeffs

    def __add__(self, other):
        coeffs = self.coeffs + other.coeffs
        return FourierCoefficients(coeffs)
    
    def __rmul__(self, other):
        coeffs = other * self.coeffs
        return FourierCoefficients(coeffs)
    
    def __abs__(self):
        coeffs = abs(self.coeffs)
        return FourierCoefficients(coeffs)

    def sum(self):
        return self.coeffs.sum(dim=[1, 2, 3])

class ShearletCoefficients(Coefficients):
    def __init__(self, coeffs, system):
        super().__init__()

        self.coeffs = coeffs
        self.system = system
    
    def clone(self):
        return ShearletCoefficients(self.get())
    
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

    def __add__(self, other):
        coeffs = self.coeffs + other.coeffs
        return ShearletCoefficients(coeffs, self.system)
    
    def __rmul__(self, other):
        coeffs = other * self.coeffs
        return ShearletCoefficients(coeffs, self.system)
    
    def __abs__(self):
        coeffs = abs(self.coeffs)
        return ShearletCoefficients(coeffs, self.system)

    def sum(self):
        return self.coeffs.sum(dim=[1, 2, 3, 4])
