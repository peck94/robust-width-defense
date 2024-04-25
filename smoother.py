import torch

class Smoother(torch.nn.Module):
    def __init__(self, model, reconstructor, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.reconstructor = reconstructor
    
    def forward(self, x):
        x_rec = self.reconstructor.generate(x.float()).float()
        return self.model(x_rec)
