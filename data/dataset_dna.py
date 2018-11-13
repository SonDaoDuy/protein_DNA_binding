import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import random
import numpy as np
import pickle
import torch

class DNADataset(DatasetBase):
	"""docstring for DNADataset"""
	def __init__(self, opt, is_for_train):
		super(DNADataset, self).__init__(opt, is_for_train)
		self._name = 'DNADataset'
		#read dataset
		self._read_dataset_paths()

	def __getitem__(self, index):
		assert (index < self._dataset_size)

		input_seq = None
		seq_label = None
		while input_seq is None or seq_label is None:
			sequence = self._seqs[index]
			input_seq = self._sequence_to_embeb(sequence)
			#seq_label = np.array(self._lbs[index])
			seq_label = self._lbs[index]

			if input_seq is None:
				print("error reading sequence number %d" % format(index))

			if seq_label is None:
				print("error reading sequence label number %d" % format(index))

		#transform data
		#in_seq =  self._transform(input_seq)
		#in_seq = torch.tensor(input_seq)


		#pack data
		sample = {'in_seq': input_seq,
				'label': seq_label
				}

		return sample

	def __len__(self):
		return self._dataset_size

	def _read_dataset_paths(self):
		self._root = self._opt.data_dir
		data_file = os.path.join(self._root, self._opt.data_file)
		data_dna, label_dna = self._read_data_file(data_file)

		#read seqs and lbs
		use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
		use_ids_filepath = os.path.join(self._root, use_ids_filename)
		ids = self._read_ids(use_ids_filepath)
		self._seqs = self._read_seqs(ids, data_dna)
		self._lbs = self._read_lbs(ids, label_dna)

		print("sequence size: %s" % format(len(self._seqs)))
		print("label size: %s" % format(len(self._lbs)))

		self._dataset_size = len(self._seqs)

	def _create_transform(self):
		transform_list = [transforms.ToTensor()]
		self._transform = transforms.Compose(transform_list)

	def _read_ids(self, use_ids_filepath):
		ids = np.loadtxt(use_ids_filepath)
		return ids

	def _read_seqs(self, ids, data_dna):
		seq = []
		for k in ids:
			seq.append(data_dna[int(k)])
		return seq

	def _read_lbs(self, ids, label_dna):
		label = []
		for k in ids:
			label.append(label_dna[int(k)])
		return label

	def _read_data_file(self, data_file):
		data_dna = []
		label_dna = []
		with open(data_file, 'rb') as f:
			data = pickle.load(f)

		for i,k in data.items():
			data_dna.append(i)
			label_dna.append(k)

		return data_dna, label_dna

	def _sequence_to_embeb(self, sequence):
		emb = []
		fill_num = 0.05
		no_of_cell = 20
		slide = self._opt.slide
		scan = self._opt.scan
		# m = 6, n = 10
		for i in range(scan - 1):
			row = [fill_num]*no_of_cell
			emb.append(row)

		for letter in sequence:
			row = self._gen_one_hot(letter)
			emb.append(row)

		for i in range(scan - 1):
			row = [fill_num]*no_of_cell
			emb.append(row)

		emb = np.array(emb)

		return emb

	def _gen_one_hot(self, letter):
		no_of_cell = 20
		row = [0]*no_of_cell
		if letter == 'A':
			row[0] = 1
		elif letter == 'V':
			row[1] = 1
		elif letter == 'K':
			row[2] = 1
		elif letter == 'I':
			row[3] = 1
		elif letter == 'S':
			row[4] = 1
		elif letter == 'Q':
			row[5] = 1
		elif letter == 'Y':
			row[6] = 1
		elif letter == 'C':
			row[7] = 1
		elif letter == 'R':
			row[8] = 1
		elif letter == 'T':
			row[9] = 1
		elif letter == 'L':
			row[10] = 1
		elif letter == 'N':
			row[11] = 1
		elif letter == 'F':
			row[12] = 1
		elif letter == 'D':
			row[13] = 1
		elif letter == 'E':
			row[14] = 1
		elif letter == 'G':
			row[15] = 1
		elif letter == 'M':
			row[16] = 1
		elif letter == 'P':
			row[17] = 1
		elif letter == 'W':
			row[18] = 1
		elif letter == 'H':
			row[19] = 1

		return row


