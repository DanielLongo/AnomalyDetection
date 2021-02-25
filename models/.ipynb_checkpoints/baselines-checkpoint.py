import torch
import torch.nn as nn
class Average(nn.Module):
    def __init__(self, n_predictions):
        super().__init__()
        
        self.n_predictions = n_predictions
    
    
    def forward(self, x):
        # single channel only
        x = x.view(x.shape[0], -1)
        sums = torch.sum(x[:, -10:], dim=1)
        length = x.shape[1]
        sums /= 10 
        preds = sums.repeat(self.n_predictions, 1).transpose(1,0)
        preds = preds.squeeze()
        return preds