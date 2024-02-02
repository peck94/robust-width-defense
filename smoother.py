import torch
import torch.nn.functional as F

from tqdm import trange

class Smoother(torch.nn.Module):
    def __init__(self, model, reconstructor, iterations=1, nb_classes=1000, verbose=False, logits=False, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.reconstructor = reconstructor
        self.iterations = iterations
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.logits = logits
    
    def forward(self, x):
        y_preds = []
        for _ in trange(self.iterations) if self.verbose else range(self.iterations):
            x_rec = self.reconstructor.generate(x.float()).float()
            y_pred = self.model(x_rec).argmax(dim=1)
            y_preds.append(y_pred)
        y_preds = torch.mode(torch.stack(y_preds), dim=0).values

        out = F.one_hot(y_preds, self.nb_classes)
        if self.logits:
            return torch.log(out)
        else:
            return out
