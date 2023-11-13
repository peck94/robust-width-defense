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
    
    def __add__(self, other):
        return DummyCoefficients(self.coeffs + other.coeffs)
    
    def __mul__(self, c):
        return DummyCoefficients(c * self.coeffs)

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
    
    def __add__(self, other):
        low = self.low + other.low
        high = [h1 + h2 for h1, h2 in zip(self.high, other.high)]
        return WaveletCoefficients((low, high))
    
    def __mul__(self, c):
        low = c * self.low
        high = [c * h for h in self.high]
        return WaveletCoefficients((low, high))

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
    
    def __add__(self, other):
        low = self.low + other.low
        high = [h1 + h2 for h1, h2 in zip(self.high, other.high)]
        return DTCWTCoefficients((low, high))
    
    def __mul__(self, c):
        low = c * self.low
        high = [c * h for h in self.high]
        return DTCWTCoefficients((low, high))

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
    
    def __add__(self, other):
        return FourierCoefficients(self.coeffs + other.coeffs)
    
    def __mul__(self, c):
        return FourierCoefficients(c * self.coeffs)

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
    
    def __add__(self, other):
        return ShearletCoefficients(self.coeffs + other.coeffs, self.system)
    
    def __mul__(self, c):
        return ShearletCoefficients(c * self.coeffs, self.system)
