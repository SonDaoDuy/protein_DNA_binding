import time
from options.train_options import TrainOptions
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from models.models import ModelsFactory
from collections import OrderedDict
import os
import pickle
from utils import util
import math

class Train:
	"""docstring for Train"""
	def __init__(self, ):
		self._opt = TrainOptions().parse()
		self._save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name)
		util.mkdirs(self._save_path)
		self._log_path_train = os.path.join(self._save_path, 'loss_log_train.txt')
		self._log_path_val = os.path.join(self._save_path, 'loss_log_val.txt')
		self._log_path_val_metric = os.path.join(self._save_path, 'loss_log_val_metric.txt')
		self._log_path_test = os.path.join(self._save_path, 'loss_log_test.txt')
		self._log_path_test_metric = os.path.join(self._save_path, 'loss_log_test_metric.txt')
		data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True, is_for_val=False)
		data_loader_val = CustomDatasetDataLoader(self._opt, is_for_train=False, is_for_val=True)
		data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False, is_for_val=False)

		self._dataset_train = data_loader_train.load_data()
		self._dataset_val = data_loader_val.load_data()
		self._dataset_test = data_loader_test.load_data()

		self._dataset_train_size = len(data_loader_train)
		self._dataset_val_size = len(data_loader_val)
		self._dataset_test_size = len(data_loader_test)
		print('#train sequences = %d' % self._dataset_train_size)
		print('#validate sequences = %d' % self._dataset_val_size)
		print('#test sequences = %d' % self._dataset_test_size)
		self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)

		self._train()

	def _train(self):
		self._total_steps = self._opt.load_epoch * self._dataset_train_size
		self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
		self._last_print_time = time.time()

		for i_epoch in range(self._opt.load_epoch + 1, self._opt.total_epoch + 1):
			epoch_start_time = time.time()

			#train epoch
			self._train_epoch(i_epoch)

			#save_model
			# print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
			# self._model.save(i_epoch)

			# print epoch info
			time_epoch = time.time() - epoch_start_time
			print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
				(i_epoch, self._opt.total_epoch, time_epoch, time_epoch / 60, time_epoch / 3600))

	def _train_epoch(self, i_epoch):
		epoch_iter = 0
		train_error = OrderedDict()
		self._model.set_train()
		for i_train_batch, train_batch in enumerate(self._dataset_train):
			iter_start_time = time.time()

			# train model
			self._model.set_input(train_batch)
			self._model.optimize_parameters()

			# update epoch info
			self._total_steps += self._opt.batch_size
			epoch_iter += self._opt.batch_size
			errors = self._model.get_current_errors()
			for k, v in errors.items():
				if k in train_error:
				    train_error[k] += v
				else:
				    train_error[k] = v

		for k in train_error.keys():
			train_error[k] /= i_train_batch
		
		# display train and val error
		t = (time.time() - iter_start_time)
		self._print_current_train_errors(i_epoch, train_error, t)
		self._display_visualizer_val(i_epoch)
		self._display_visualizer_test(i_epoch)
		#self._last_print_time = time.time()

		# save model
		print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
		self._model.save(i_epoch)
		#self._last_save_latest_time = time.time()

	def _display_visualizer_test(self, i_epoch):
		val_start_time = time.time()

		#set model to eval mode
		self._model.set_eval()

		#evaluate
		val_error = OrderedDict()
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		for i_val_batch, val_batch in enumerate(self._dataset_test):
			self._model.set_input(val_batch)
			tp_batch, fp_batch, tn_batch, fn_batch = self._model.forward()
			tp += tp_batch
			fp += fp_batch
			tn += tn_batch
			fn += fn_batch
			errors = self._model.get_current_errors()
			# store current batch errors
			for k, v in errors.items():
				if k in val_error:
					val_error[k] += v
				else:
					val_error[k] = v

		for k in val_error.keys():
			val_error[k] /= i_val_batch

		sen, spe, acc, pre, mcc = self._cal_metric(tp, fp, tn, fn)

		#visualize
		#print and save val loss
		t = (time.time() - val_start_time)
		self._print_current_test_errors(i_epoch, val_error, t)
		self._print_current_test_metrics(i_epoch, sen, spe, acc, pre, mcc)
		#set back to train
		self._model.set_train()
		
	def _display_visualizer_val(self, i_epoch):
		val_start_time = time.time()

		#set model to eval mode
		self._model.set_eval()

		#evaluate
		val_error = OrderedDict()
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		for i_val_batch, val_batch in enumerate(self._dataset_val):
			self._model.set_input(val_batch)
			tp_batch, fp_batch, tn_batch, fn_batch = self._model.forward()
			tp += tp_batch
			fp += fp_batch
			tn += tn_batch
			fn += fn_batch
			errors = self._model.get_current_errors()
			# store current batch errors
			for k, v in errors.items():
				if k in val_error:
					val_error[k] += v
				else:
					val_error[k] = v

		for k in val_error.keys():
			val_error[k] /= i_val_batch

		print("tp: %s\n" % format(tp))
		print("fp: %s\n" % format(fp))
		print("tn: %s\n" % format(tn))
		print("fn: %s\n" % format(fn))
		sen, spe, acc, pre, mcc = self._cal_metric(tp, fp, tn, fn)

		#visualize
		#print and save val loss
		t = (time.time() - val_start_time)
		self._print_current_validate_errors(i_epoch, val_error, t)
		self._print_current_validate_metrics(i_epoch, sen, spe, acc, pre, mcc)
		#set back to train
		self._model.set_train()

	def _cal_metric(self, tp, fp, tn, fn):
		sen = tp/(tp+fn+0.00001)
		spe = tn/(tn+fp+0.00001)
		acc = (tp + tn)/(tp+fn+tn+fp+0.00001)
		pre = tp/(tp+fp+0.00001)
		mcc = (tp*tn - fn*fp)/(math.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))+0.00001)

		return sen, spe, acc, pre, mcc

	def _print_current_train_errors(self, epoch, errors, t):
		log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
		message = '%s, epoch: %d, t/smpl: %.3fs, ' % (log_time, epoch, t)
		for k, v in errors.items():
			message += '%s: %.3f ' % (k, v)

		print(message)
		with open(self._log_path_train, "a") as log_file:
			log_file.write('%s\n' % message)

	def _print_current_validate_errors(self, epoch, errors, t):
		log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
		message = '%s, V, epoch: %d, time_to_val: %ds, ' % (log_time, epoch, t)
		for k, v in errors.items():
			message += '%s: %.3f ' % (k, v)

		print(message)
		with open(self._log_path_val, "a") as log_file:
			log_file.write('%s\n' % message)

	def _print_current_validate_metrics(self, epoch, sen, spe, acc, pre, mcc):
		log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
		message = '%s, V, epoch: %d, sen: %.3f, spe: %.3f, acc: %.3f, pre: %.3f, mcc: %.3f' % (log_time, epoch, sen, spe, acc, pre, mcc)
		print(message)
		with open(self._log_path_val_metric, "a") as log_file:
			log_file.write('%s\n' % message)

	def _print_current_test_errors(self, epoch, errors, t):
		log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
		message = '%s, Test, epoch: %d, time_to_val: %ds, ' % (log_time, epoch, t)
		for k, v in errors.items():
			message += '%s: %.3f ' % (k, v)

		print(message)
		with open(self._log_path_test, "a") as log_file:
			log_file.write('%s\n' % message)

	def _print_current_test_metrics(self, epoch, sen, spe, acc, pre, mcc):
		log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
		message = '%s, Test, epoch: %d, sen: %.3f, spe: %.3f, acc: %.3f, pre: %.3f, mcc: %.3f' % (log_time, epoch, sen, spe, acc, pre, mcc)
		print(message)
		with open(self._log_path_test_metric, "a") as log_file:
			log_file.write('%s\n' % message)

def main():
	# _opt = TrainOptions().parse()
	# file_data = './sample_dataset/data_dna.txt'
	# #file_data = './sample_dataset/data_dna.pkl'
	# file_train = './sample_dataset/train_ids.csv'
	# file_val = './sample_dataset/val_ids.csv'
	# split = 5
	# seed = _opt.seed
	# data_size = 0
	# with open(file_data, 'r') as f:
	# 	for line in f:
	# 		data_size += 1

	# #print("len of data_dna")
	# fold_size = int(data_size/5)
	
	# hold_data = [k for k in range(0,data_size)]
	# val_part = [k for k in range(seed*fold_size, (seed+1)*fold_size)]
	# train_part = list(set(hold_data) - set(val_part))
	# with open(file_train, 'w') as f:
	# 	for item in train_part:
	# 		f.write("%s\n" % item)

	# with open(file_val, 'w') as f:
	# 	for item in val_part:
	# 		f.write("%s\n" % item)

	Train()


if __name__ == '__main__':
	main()



