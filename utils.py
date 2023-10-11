import torch

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

def hard_thresh(xs, lam):
    if isinstance(xs, tuple):
        return (hard_thresh(x, lam) for x in xs)
    elif isinstance(xs, list):
        return [hard_thresh(x, lam) for x in xs]
    else:
        return torch.where(abs(xs) < lam, torch.zeros_like(xs), xs)

def soft_thresh(xs, lam):
    if isinstance(xs, tuple):
        return (soft_thresh(x, lam) for x in xs)
    elif isinstance(xs, list):
        return [soft_thresh(x, lam) for x in xs]
    else:
        return torch.zeros_like(xs) + (abs(xs) - lam) / abs(xs) * xs * (abs(xs) > lam)
