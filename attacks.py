from autoattack import AutoAttack as AA

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
        elif args.attack == 'apgd':
            self.adversary = AA(target, norm=args.norm, eps=args.eps/255, version='custom', attacks_to_run=['apgd-ce'])
        elif args.attack == 'full':
            self.adversary = AA(target, norm=args.norm, eps=args.eps/255, version='rand')
    
    def generate(self, x_batch, y_batch):
        return self.adversary.run_standard_evaluation(x_batch.detach().cpu(), y_batch.detach().cpu(), bs=self.args.bs)
