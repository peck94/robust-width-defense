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

        if args.adapt:
            self.adversary = AA(defense, norm=args.norm, eps=args.eps/255, version='rand')
            self.adversary.attacks_to_run = ['square', 'apgd-ce']
        else:
            self.adversary = AA(model, norm=args.norm, eps=args.eps/255, version='standard')
    
    def generate(self, x_batch, y_batch):
        return self.adversary.run_standard_evaluation(x_batch.detach().cpu(), y_batch.detach().cpu(), bs=self.args.bs)
