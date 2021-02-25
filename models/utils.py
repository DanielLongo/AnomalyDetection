import torch
import math
from dynaconf import settings


def preds_with_sliding_window(x, num_inputs, num_preds, model):
    if len(x.shape) == 2:
        # this is a single example rather than batch
        x = x.view(x.shape[0], 1, x.shape[1])

    assert (len(x.shape) == 3), "Preds with sliding window takes shape (m, n, t)"

    output_length = settings.N_PREDICTIONS
    num_iters = math.ceil(num_preds / output_length)

    preds = []
    model = model.eval()
    with torch.no_grad():
        for i in range(num_iters):
            start = i * output_length
            end = start + num_inputs
            cur_preds = model(x[:, start:end, :])
            print(cur_preds.shape)
            preds.append(cur_preds.cpu())
    preds = torch.cat(preds, dim=1)
    return preds[:, :num_preds]


def check_model_is_cuda(model):
    try:
        return next(model.parameters()).is_cuda
    except StopIteration:
        # model has no params
        return True
