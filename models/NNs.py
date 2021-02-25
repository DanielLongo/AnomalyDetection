import torch
import torch.nn as nn
class TorchNN(nn.Module):
    def __init__(self, input_length=250,n_predictions=10):
        super().__init__()
        self.fc1 = nn.Linear(input_length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_predictions)
    
    def forward(self, input_seq):
        input_seq = input_seq.squeeze()
        h = torch.nn.functional.relu(self.fc1(input_seq))
        h = torch.nn.functional.relu(self.fc2(h))
        h = torch.nn.functional.relu(self.fc3(h))
        out = self.fc4(h)
        return out
