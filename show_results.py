import matplotlib.pyplot as plt
import numpy as np
import os

def read_data_from_files():
	file_train_loss = './checkpoints/exp12/loss_log_train.txt'
	file_val_loss = './checkpoints/exp12/loss_log_val.txt'
	file_metric = './checkpoints/exp12/loss_log_val_metric.txt'

	all_train_loss = []
	with open(file_train_loss, 'r') as f:
		for line in f:
			#line = f.readline()
			loss = float(line.split(' ')[7])
			all_train_loss.append(loss)

	all_val_loss = []
	with open(file_val_loss, 'r') as f:
		for line in f:
			loss = float(line.split(' ')[8])
			all_val_loss.append(loss)
	    
	sen = []
	pre = []
	acc = []
	spe = []
	mcc = []
	with open(file_metric, 'r') as f:
		for line in f:
			item = line.split(' ')
			sen_i = float(item[6][:-1])
			sen.append(sen_i)
			spe_i = float(item[8][:-1])
			spe.append(spe_i)
			acc_i = float(item[10][:-1])
			acc.append(acc_i)
			pre_i = float(item[12][:-1])
			pre.append(pre_i)
			mcc_i = float(item[14])
			mcc.append(mcc_i)

	return all_train_loss, all_val_loss, sen, pre, acc, spe, mcc

def draw_graph(all_train_loss, all_val_loss, sen, pre, acc, spe, mcc):
	epochs = np.arange(len(acc))

	fig = plt.figure()
	#plot loss
	bx = plt.subplot(211)
	bx.plot(epochs, all_train_loss, label='train_loss')
	bx.plot(epochs, all_val_loss, label='val_loss')
	# Shrink current axis by 20%
	box = bx.get_position()
	bx.set_position([box.x0, box.y0, box.width, box.height])

	# Put a legend to the right of the current axis
	bx.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	#plot metric
	ax = plt.subplot(212)

	ax.plot(epochs, sen, label='sen')
	ax.plot(epochs, spe, label='spe')
	ax.plot(epochs, acc, label='acc')
	ax.plot(epochs, pre, label='pre')
	ax.plot(epochs, mcc, label='mcc')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.show()

def main():
	all_train_loss, all_val_loss, sen, pre, acc, spe, mcc = read_data_from_files()
	draw_graph(all_train_loss, all_val_loss, sen, pre, acc, spe, mcc)

if __name__ == '__main__':
	main()