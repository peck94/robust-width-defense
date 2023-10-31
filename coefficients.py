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
        b = self.coeffs.shape[-1]
        p = (b - 1) // 2

        bands = self.coeffs[:, :, :, :, p:b]
        n = self.coeffs.shape[0]
        c = self.coeffs.shape[1]

        s_hats = torch.zeros(n, bands.shape[-1])
        for i in range(bands.shape[-1]):
            s_hats[:, i] = torch.median(abs(bands[:, :, :, :, i].view(n, -1)), dim=1).values
        s_hat = (alpha * torch.median(s_hats, dim=1).values / 0.6745).to(self.coeffs.device)

        s_y = torch.sqrt(torch.var(self.coeffs, dim=(1, 2, 3, 4)))
        s_x = torch.sqrt(torch.maximum(torch.zeros_like(s_y), torch.square(s_y) - torch.square(s_hat)))

        upper = abs(self.coeffs).view(n, -1).max(dim=1).values
        tau = torch.minimum(torch.square(s_hat) / s_x, upper)
        return tau.reshape(n, 1, 1, 1, 1).expand(-1, c, -1, -1, -1)

    def hard_thresh(self, alpha):
        self.coeffs = self.ht(self.coeffs, alpha)
    
    def soft_thresh(self, alpha):
        self.coeffs = self.st(self.coeffs, alpha)

    def get(self):
        return self.coeffs
