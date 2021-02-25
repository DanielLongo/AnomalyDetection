import torch

def preds_with_sliding_window(x, num_preds,  model):
	if len(x.shape) == 2:
		# this is a single example rather than batch
		x = x.view(x.shape[0], 1, x.shape[1])
	assert(len(x.shape) == 3)
	
	output_length = model.
