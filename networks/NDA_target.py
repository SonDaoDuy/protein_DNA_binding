import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class Network(NetworkBase):
	"""docstring for Network
	n = 10
	m = 6
	"""
	def __init__(self, conv_dim = 22, input_dim = 1, output_dim = 2):
		super(Network, self).__init__()
		self._name = 'NDA_target'
		self.feat_dim = conv_dim

		layers = []
		layers.append(nn.Conv2d(input_dim, conv_dim, (6, 20), 1, 0)) #B*1*21*20 -> B*conv_dim*16*1
		layers.append(nn.BatchNorm2d(conv_dim))
		layers.append(nn.ReLU())
		layers.append(nn.AvgPool2d((16,1), stride=(1,1))) #B*conv_dim*16*1 -> B*conv_dim*1*1
		
		self.embed = nn.Sequential(*layers)
		self.fc = nn.Linear(conv_dim, output_dim)
		self.final = nn.Softmax()

	def forward(self, input):
		x = torch.unsqueeze(input, 1)
		# print(x.size())
		x = self.embed(x)
		x = x.view(-1, self.feat_dim)
		x = self.fc(x)
		x = self.final(x)

		return x