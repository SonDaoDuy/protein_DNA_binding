from Bio import SeqIO
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

def main():
	data_file = './PDNA-543_sequence.fasta'
	label_file = './PDNA-543_label.fasta'
	data_file_test = './PDNA-TEST_sequence.fasta'
	label_file_test = './PDNA-TEST_label.fasta'

	train_data_file = './sample_dataset/train_data_dna.txt'
	train_label_file = './sample_dataset/train_label_dna.txt'
	val_data_file = './sample_dataset/val_data_dna.txt'
	val_label_file = './sample_dataset/val_label_dna.txt'
	test_data_file = './sample_dataset/test_data_dna.txt'
	test_label_file = './sample_dataset/test_label_dna.txt'

	data = []
	#get dnd seq
	count = 0
	size = 5
	for record in  SeqIO.parse(data_file, "fasta"):
		index = 0
		while index < len(record.seq):
			#print(index)
			if index < size:
				seq_to_save = record.seq[0:(index+size+1)]
				for i in range(size-index):
					seq_to_save = 'X' + seq_to_save
				data.append(seq_to_save)
			elif (index + size) >= len(record.seq):
				seq_to_save = record.seq[(index-size):len(record.seq)]
				for i in range(index + size + 1 - len(record.seq)):
					seq_to_save += 'X'
				data.append(seq_to_save)
			else:
				data.append(record.seq[(index-size):(index+size+1)])

			index += 1
			count += 1


	label = []
	#get seq label
	count = 0
	for record in  SeqIO.parse(label_file, "fasta"):
		index = 0
		for i in record.seq:
			item = int(i)
			label.append(item)
			index += 1
			count += 1


	#save to pkl file
	print(len(data))
	print(len(label))

	x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2)
	with open(train_data_file, 'w') as f:
		for i in range(len(x_train)):
			f.write("%s\n" % format(x_train[i]))
	with open(train_label_file, 'w') as f:
		for i in range(len(y_train)):
			f.write("%s\n" % format(y_train[i]))

	with open(val_data_file, 'w') as f:
		for i in range(len(x_val)):
			f.write("%s\n" % format(x_val[i]))
	with open(val_label_file, 'w') as f:
		for i in range(len(y_val)):
			f.write("%s\n" % format(y_val[i]))

	data_test = []
	#get dnd seq
	count = 0
	size = 5
	for record in  SeqIO.parse(data_file_test, "fasta"):
		index = 0
		while index < len(record.seq):
			#print(index)
			if index < size:
				seq_to_save = record.seq[0:(index+size+1)]
				for i in range(size-index):
					seq_to_save = 'X' + seq_to_save
				data_test.append(seq_to_save)
			elif (index + size) >= len(record.seq):
				seq_to_save = record.seq[(index-size):len(record.seq)]
				for i in range(index + size + 1 - len(record.seq)):
					seq_to_save += 'X'
				data_test.append(seq_to_save)
			else:
				data_test.append(record.seq[(index-size):(index+size+1)])

			index += 1
			count += 1


	label_test = []
	#get seq label
	count = 0
	for record in  SeqIO.parse(label_file_test, "fasta"):
		index = 0
		for i in record.seq:
			item = int(i)
			label_test.append(item)
			index += 1
			count += 1


	#save to pkl file
	print(len(data_test))
	print(len(label_test))

	with open(test_data_file, 'w') as f:
		for i in range(len(data_test)):
			f.write("%s\n" % format(data_test[i]))
	with open(test_label_file, 'w') as f:
		for i in range(len(label_test)):
			f.write("%s\n" % format(label_test[i]))


if __name__ == '__main__':
	main()