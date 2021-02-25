import torch
import torch.nn as nn

# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=80, num_layers=2, output_size=1, n_predictions=80):
#         super().__init__()

#         self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=.3)

#         self.linear_layers = nn.Sequential(
#             nn.Linear(hidden_layer_size, hidden_layer_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_layer_size // 2, n_predictions)
#         ) 
#         self.m = nn.Sigmoid()

#     def forward(self, input_seq):
#         _, (h_n,_) = self.lstm(input_seq)
#         h_n = h_n[-1] # from last layer
#         h_n = h_n.squeeze()
#         predictions = self.linear_layers(h_n)
#         return predictions

class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_layer_size=80, num_layers=2, output_size=1, n_predictions=80, dropout=.3):
		super().__init__()
		self.output_size = output_size
		self.n_predictions = n_predictions
		self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout)

		self.fc1 = nn.Linear(hidden_layer_size, n_predictions * output_size)


	def forward(self, input_seq):
		_, (h_n,_) = self.lstm(input_seq)
		h_n = h_n[-1] # from last layer
		h_n = h_n.squeeze()
		preds = self.fc1(h_n).view(-1, self.n_predictions, self.output_size)
		return preds
