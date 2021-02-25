import torch
from progress.bar import ChargingBar
from dynaconf import settings

def get_pred_diff(model, full_recording, criterion, use_cuda=True):
    if use_cuda:
        full_recording = full_recording.cuda()
        
    full_recording = full_recording.unsqueeze(2)
    partial_recording = full_recording[:, :-settings.N_PREDICTIONS]
    
    with torch.no_grad():
        pred = model(partial_recording).squeeze()
    end_recording = full_recording[:, -settings.N_PREDICTIONS:].squeeze()
    diff = criterion(end_recording, pred)
    return diff

def get_pred_diffs(model, dataset, criterion, use_cuda=True):
    diffs = []
    bar = ChargingBar('Calculating', max=len(dataset))
    for example in dataset:
        diffs.append(get_pred_diff(model, example, criterion , use_cuda=use_cuda).item())
        bar.next()
    bar.finish()
    return diffs