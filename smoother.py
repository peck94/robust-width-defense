import torch
import torch.nn.functional as F

from tqdm import trange

class Smoother(torch.nn.Module):
    def __init__(self, model, reconstructor, iterations=1, nb_classes=1000, verbose=False, softmax=False, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.reconstructor = reconstructor
        self.iterations = iterations
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.softmax = softmax
    
    def forward(self, x):
        y_out = torch.zeros(x.shape[0], self.nb_classes).to(x.device)
        for _ in trange(self.iterations) if self.verbose else range(self.iterations):
            x_rec = self.reconstructor.generate(x.float()).float()
            y_pred = self.model(x_rec)
            y_out = y_out + y_pred
        y_out = y_out / self.iterations

        if self.softmax:
            return F.softmax(y_out, dim=1)
        else:
            return y_out
