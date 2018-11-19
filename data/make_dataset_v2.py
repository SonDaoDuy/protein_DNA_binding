from Bio import SeqIO
import numpy as np
import pickle
import os

def main():
	data_file = './PDNA-543_sequence.fasta'
	label_file = './PDNA-543_label.fasta'
	# dna_data_file = './sample_dataset/data_dna_v2.pkl'
	data_file_save = './sample_dataset/data_dna.txt'
	label_file_save = './sample_dataset/label_dna.txt'
	
	data = []
	#get dnd seq
	count = 0
	size = 4
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
		# 	if count == 100:
		# 		break

		# break


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
		# 	if count == 100:
		# 		break
		# break

	#save to pkl file
	packed_data = dict()
	# for i in range(size):
	# 	print(data[i])
	# 	print(label[i])
	# 	print(data[99-i])
	# 	print(label[99-i])
	print(len(data))
	print(len(label))
	with open(data_file_save, 'w') as f:
		for i in range(len(data)):
			f.write("%s\n" % format(data[i]))
	with open(label_file_save, 'w') as f:
		for i in range(len(label)):
			f.write("%s\n" % format(label[i]))

if __name__ == '__main__':
	main()