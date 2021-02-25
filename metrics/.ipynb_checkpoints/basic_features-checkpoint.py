import torch
def mse(value, pred):
    diff = torch.sum((pred - value) ** 2)
    if torch.sum(abs(value)) != 0:
        diff /= torch.sum(abs(value))
    return diff