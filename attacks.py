import torch
import torch.nn.functional as F

from autoattack import AutoAttack as AA

from art.attacks.evasion import SimBA as SA
from art.estimators.classification import PyTorchClassifier

class Attack:
    def __init__(self, args, model, defense):
        self.args = args
        self.model = model
        self.defense = defense
    
    def generate(self, x_batch, y_batch):
        return x_batch

class AutoAttack(Attack):
    def __init__(self, args, model, defense):
        super().__init__(args, model, defense)

        target = model
        if args.adapt:
            target = defense

        if args.attack == 'square':
            self.adversary = AA(target, norm=args.norm, eps=args.eps/255, version='custom', attacks_to_run=['square'])
            self.adversary.square.n_queries = 500
        elif args.attack == 'apgd':
            self.adversary = AA(target, norm=args.norm, eps=args.eps/255, version='custom', attacks_to_run=['apgd-ce'])
    
    def generate(self, x_batch, y_batch):
        return self.adversary.run_standard_evaluation(x_batch.detach().cpu(), y_batch.detach().cpu(), bs=self.args.bs)
