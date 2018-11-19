import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class Networkv3(NetworkBase):
	"""docstring for Networkv3
	slide = 11
	seg_size = 20
	input_dim = slide * seq_size
	"""
	def __init__(self, input_dim = 11*20, output_dim = 2):
		super(Networkv3, self).__init__()
		self._name = 'NDA_target_v3'
		self.hidden_dim = 32

		layers = []
		layers.append(nn.Linear(input_dim, self.hidden_dim))
		layers.append(nn.Sigmoid())
		layers.append(nn.Dropout())
		layers.append(nn.Linear(self.hidden_dim, output_dim))
		#layers.append(nn.Softmax())
		self.fc = nn.Sequential(*layers)

	def forward(self, input):
		x = torch.unsqueeze(input, 1)
		x = x.view(-1, x.size(2)*x.size(3))
		# print(x.size())
		x = self.fc(x)

		return x