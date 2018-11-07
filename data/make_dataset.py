from Bio import SeqIO
import numpy as np
import pickle
import os

def main():
	data_file = 'D:\\TargetDNA\\PDNA-543_sequence.fasta'
	label_file = 'D:\\TargetDNA\\PDNA-543_label.fasta'
	dna_data_file = 'D:\\TargetDNA\\sample_dataset\\data_dna.pkl'
	
	data = []
	#get dnd seq
	count = 0
	for record in  SeqIO.parse(data_file, "fasta"):
		index = 0
		while index+10 <= len(record.seq):
			#print(index)
			data.append(record.seq[index:(index+10)])
			# print(data[count])
			index += 1
			count += 1
			# if count == 100:
			# 	break

		#break

	label = []
	#get seq label
	count = 0
	for record in  SeqIO.parse(label_file, "fasta"):
		index = 0
		while index+10 <= len(record.seq):
			item = [int(k) for k in record.seq[index:index+10]]
			label.append(item)
			index += 1
			count += 1
		# 	if count == 100:
		# 		break

		# break

	#save to pkl file
	packed_data = dict()
	print(len(data))
	for i in range(len(data)):
		#print(i)
		packed_data[data[i]] = label[i]

	with open(dna_data_file, 'wb') as f:
		pickle.dump(packed_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()