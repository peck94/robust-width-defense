import torch

import numpy as np

import json

from pathlib import Path

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
    
    @property
    def sem(self):
        _, sigma2 = self.values
        return 1.96 * np.sqrt(sigma2 / self.count)
    
    def to_json(self):
        return {
            'mean': self.mean,
            'count': self.count,
            'M2': self.M2
        }
    
    def from_json(self, data):
        self.count = data['count']
        self.mean = data['mean']
        self.M2 = data['M2']
        return self

class Logger:
    def __init__(self, location):
        self.location = Path(location)
        self.data = []

        if self.exists():
            self.load()
    
    def exists(self):
        return self.location.exists()
    
    def load(self):
        with open(self.location, 'r') as f:
            self.data = json.load(f)
    
    def save(self):
        with open(self.location, 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=4)
    
    def find_experiment(self, args):
        for i, item in enumerate(self.data):
            if item['eps'] == args.eps and item['norm'] == args.norm:
                return i
        return -1
    
    def get_experiment(self, args):
        orig_acc, adv_acc = Welford(), Welford()
        index = self.find_experiment(args)
        if index >= 0:
            item = self.data[index]
            orig_acc.from_json(item['orig_acc'])
            adv_acc.from_json(item['adv_acc'])

        return orig_acc, adv_acc
    
    def get_experiments(self, sort=False, norm='Linf'):
        experiments = []
        for item in self.data:
            if item['norm'] == norm:
                experiments.append({
                    'eps': item['eps'],
                    'norm': item['norm'],
                    'orig_acc': Welford().from_json(item['orig_acc']),
                    'adv_acc': Welford().from_json(item['adv_acc'])
                })
        if sort:
            experiments = sorted(experiments, key=lambda item: item['eps'])
        return experiments
    
    def set_experiment(self, args, orig_acc, adv_acc):
        item = {
                'eps': args.eps,
                'norm': args.norm,
                'orig_acc': orig_acc.to_json(),
                'adv_acc': adv_acc.to_json()
        }

        index = self.find_experiment(args)
        if index >= 0:
            self.data[index] = item
        else:
            self.data.append(item)
        self.save()

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
