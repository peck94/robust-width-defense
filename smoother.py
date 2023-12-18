import torch
import torch.nn.functional as F

class Smoother(torch.nn.Module):
    def __init__(self, model, reconstructor, iterations=1, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.reconstructor = reconstructor
        self.iterations = iterations
    
    def forward(self, x):
        y_preds = []
        for _ in range(self.iterations):
            x_rec = self.reconstructor.generate(x.float()).float()
            y_pred = self.model(x_rec).argmax(dim=1)
            y_preds.append(y_pred)
        y_preds = torch.mode(torch.stack(y_preds), dim=0).values
        
        nb_classes = y_pred.shape[-1]
        preds = F.one_hot(y_preds, nb_classes)
        print(preds)

        return preds
