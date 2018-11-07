import torch
from collections import OrderedDict
from torch.autograd import Variable
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np

class DNAmodel(BaseModel):
	"""docstring for DNAmodel"""
	def __init__(self, opt):
		super(DNAmodel, self).__init__(opt)
		self._name = 'DNA_model'

		#create network
		self._init_create_networks()

		#init train vars
		if self._is_train:
			self._init_train_vars()

		#load netwroks and optimizers
		if not self._is_train or self._opt.load_epoch > 0:
			self.load()

		#prefetch vars
		self._init_prefetch_inputs()

		#init losses
		self._init_losses()

	def _init_create_networks(self):
		self._net = self._create_network('DNA_target')
		self._net.init_weights()

		if len(self._gpu_ids) > 1:
			self._net = torch.nn.DataParallel(self._net, device_ids = self._gpu_ids)
		if len(self._gpu_ids) > 0:
			self._net.cuda()

	def _create_network(self, net_name):
		return NetworksFactory.get_by_name(net_name)

	def _init_train_vars(self):
		self._current_lr_net = self._opt.lr_net

		#init optimizer
		self._optimizer_net = torch.optim.SGD(self._net.parameters(), lr=0.1, momentum=0.9)
		# self._optimizer_net = torch.optim.Adam(self._net.parameters(), lr=self._current_lr_net,
		# 	betas=[self._opt.adam_b1, self._opt.adam_b2])

	def _init_prefetch_inputs(self):
		self._input_sequence = self._Tensor(self._opt.batch_size, 1, self._opt.seq_size, self._opt.seq_size)
		self._label = self._Tensor(self._opt.batch_size, self._opt.seq_size)

	def _init_losses(self):
		#define loss function
		if len(self._gpu_ids) > 0:
			#self._criterion_net = torch.nn.MSELoss().cuda()
			self._criterion_net = torch.nn.BCELoss().cuda()
		else:
			self._criterion_net = torch.nn.BCELoss()
			#self._criterion_net = torch.nn.MSELoss()

		#define loss
		self._net_loss = Variable(self._Tensor([0]))

	def set_input(self, input):
		self._input_sequence.resize_(input['in_seq'].size()).copy_(input['in_seq'])
		self._label.resize_(input['label'].size()).copy_(input['label'])

		if len(self._gpu_ids) > 0:
			self._input_sequence = self._input_sequence.cuda(self._gpu_ids[0], async=True)
			self._label = self._label.cuda(self._gpu_ids[0], async=True)

	def set_train(self):
		self._net.train()
		self._is_train = True

	def set_eval(self):
		self._net.eval()
		self._is_train = False

	def forward(self):
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		if not self._is_train:
			#convert sequence to variables
			input_seq = Variable(self._input_sequence, volatile=True)
			label = Variable(self._label, volatile=True)

			#go through net
			predict = self._net(input_seq)

			#return something here for validation
			self._net_loss = self._criterion_net(predict, label)
			
			#metric calcalate
			for i in range(predict.size(0)):
				for j in range(predict.size(1)):
					if predict[i][j] < self._opt.threshold:
						if label[i][j] == 0:
							tn += 1
						else:
							fn += 1
					else:
						if label[i][j] == 0:
							fp += 1
						else:
							tp += 1

			return tp, fp, tn, fn

	def optimize_parameters(self):
		if self._is_train:
			#convert tensor to variables
			self._B = self._input_sequence.size(0)
			self._input_seq =  Variable(self._input_sequence)
			self._in_label = Variable(self._label)

			#train net
			loss = self._forward_net()
			self._optimizer_net.zero_grad()
			loss.backward()
			self._optimizer_net.step()

	def _forward_net(self):
		predict_lb = self._net(self._input_seq)
		self._net_loss = self._criterion_net(predict_lb, self._in_label)
		return self._net_loss

	def get_current_errors(self):
		loss_dict = OrderedDict([
			('net_loss', self._net_loss.data[0])])

		return loss_dict

	def get_current_scalars(self):
		return OrderedDict([('lr', self._current_lr_net)])

	def save(self, label):
		# save networks
		self._save_network(self._net, 'DNA', label)

		# save optimizers
		self._save_optimizer(self._optimizer_net, 'DNA', label)

	def load(self):
		load_epoch = self._opt.load_epoch

		# load net
		self._load_network(self._net, 'DNA', load_epoch)

		if self._is_train:
			# load optimizers
			self._load_optimizer(self._optimizer_net, 'DNA', load_epoch)
