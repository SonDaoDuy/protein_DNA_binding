import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class Networkv2(NetworkBase):
	"""docstring for Networkv2
	n = slide
	m = scan
	exp7: conv_dim = 8, avgpool (10,1), Linear(feat_dim*2, out_dim)
	exp3: conv_dim = 10, avgpool(11,1), Linear(feat_dim, out_dim)
	exp8: no compensasion emb, conv_dim=10, avgpool (8,1), Linear(feat_dim, out_dim)
	"""
	def __init__(self, conv_dim = 10, input_dim = 1, output_dim = 2):
		super(Networkv2, self).__init__()
		self._name = 'NDA_target_v2'
		self.feat_dim = conv_dim

		layers = []
		layers.append(nn.Conv2d(input_dim, conv_dim, (11, 20), 1, 0)) #B*1*21*20 -> B*conv_dim*11*1
		layers.append(nn.BatchNorm2d(conv_dim))
		layers.append(nn.ReLU())
		layers.append(nn.AvgPool2d((11,1), stride=(1,1))) #B*conv_dim*11*1 -> B*conv_dim*1*1
		layers.append(nn.Dropout2d())
		
		self.embed = nn.Sequential(*layers)

		layers = []
		layers.append(nn.Linear(conv_dim, output_dim))
		layers.append(nn.Sigmoid())
		self.fc = nn.Sequential(*layers)

	def forward(self, input):
		x = torch.unsqueeze(input, 1)
		# print(x.size())
		x = self.embed(x)
		x = x.view(-1, self.feat_dim)
		x = self.fc(x)

		return x