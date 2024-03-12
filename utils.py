import torch

import numpy as np

import json

class Welford:
    def __init__(self):
        self.mean = 0.0
        self.count = 0
        self.M2 = 0.0
    
    def update(self, new_value):
        self.count += 1

        delta = float(new_value) - self.mean
        self.mean += delta / self.count

        delta2 = float(new_value) - self.mean
        self.M2 += delta * delta2
    
    def update_all(self, new_values):
        for new_value in new_values:
            self.update(new_value)
    
    @property
    def values(self):
        return self.mean, self.M2 / (self.count - 1) if self.count > 1 else np.nan
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def from_json(self, data):
        obj = json.loads(data)
        self.count = obj['count']
        self.mean = obj['mean']
        self.M2 = obj['M2']

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
