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
	size = 20
	for record in  SeqIO.parse(data_file, "fasta"):
		index = 0
		while index+size <= len(record.seq):
			#print(index)
			data.append(record.seq[index:(index+size)])
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
		while index+size <= len(record.seq):
			item = [int(k) for k in record.seq[index:index+size]]
			label.append(item)
			index += 1
			count += 1
		# 	if count == 100:
		# 		break

		# break

	#save to pkl file
	packed_data = dict()
	print(len(data))
	same_seq = 0
	diff_lb = 0
	for i in range(len(data)):
		#print(i)
		if data[i] in packed_data:
			same_seq += 1
			print("seq: %s" % format(data[i]))
			if packed_data[data[i]] != label[i]:
				diff_lb += 1
				print("exist lb: %s" % format(packed_data[data[i]]))
				print("new lb: %s" % format(label[i]))
		packed_data[data[i]] = label[i]
	# for i in range(len(data)):
	# 	#print(i)
	# 	packed_data[data[i]] = label[i]

	print("num of same seq: %d" % same_seq)
	print("num of diff lb with same seq: %d" % diff_lb)
	with open(dna_data_file, 'wb') as f:
		pickle.dump(packed_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()