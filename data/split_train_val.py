import os
import numpy as np
import pickle

def main():
	file_data = 'D:\\TargetDNA\\sample_dataset\\data_dna.pkl'
	file_train = 'D:\\TargetDNA\\sample_dataset\\train_ids.csv'
	file_val = 'D:\\TargetDNA\\sample_dataset\\val_ids.csv'
	seed = 0
	split = 5

	#read pkkl file
	with open(file_data, 'rb') as f:
		data = pickle.load(f)

	data_size = len(data)
	fold_size = int(data_size/5)
	hold_data = [k for k in range(0,data_size)]
	val_part = [k for k in range(seed*fold_size, (seed+1)*fold_size)]
	print(np.array(val_part))
	train_part = list(set(hold_data) - set(val_part))

	with open(file_train, 'w') as f:
		for item in train_part:
			f.write("%s\n" % item)

	with open(file_val, 'w') as f:
		for item in val_part:
			f.write("%s\n" % item)

if __name__ == '__main__':
	main()

