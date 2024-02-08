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

        if args.adapt:
            self.adversary = AA(defense, norm=args.norm, eps=args.eps/255, version='rand')
            self.adversary.attacks_to_run = ['square', 'apgd-ce']
        else:
            self.adversary = AA(model, norm=args.norm, eps=args.eps/255, version='standard')
    
    def generate(self, x_batch, y_batch):
        return self.adversary.run_standard_evaluation(x_batch.detach().cpu(), y_batch.detach().cpu(), bs=self.args.bs)

class SimBA(Attack):
    def __init__(self, args, model, defense):
        super().__init__(args, model, defense)

        if args.adapt:
            estimator = PyTorchClassifier(
                model=defense,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=[3, 224, 224],
                nb_classes=1000)
        else:
            estimator = PyTorchClassifier(model,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=[3, 224, 224],
                nb_classes=1000)
        
        self.adversary = SA(estimator, max_iter=100)
    
    def generate(self, x_batch, y_batch):
        x_adv = self.adversary.generate(x_batch.detach().cpu().numpy(),
                                       F.one_hot(y_batch, 1000).detach().cpu().numpy())
        return torch.from_numpy(x_adv)
