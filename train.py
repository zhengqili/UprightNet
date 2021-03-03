from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.train_options import TrainOptions
from data.data_loader import *
from models.models import create_model
import random
from tensorboardX import SummaryWriter
import sys


TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

print(opt.mode)

root = '/'

summary_name = opt.dataset + '_runs/' + \
			'%s_exp'%opt.log_comment + '_net_' + opt.mode + '_lr_' + \
			str(opt.lr) + '_w_pose_' + str(opt.w_pose) + \
			'_w_grad_' + str(opt.w_grad) + \
			'_backprop_eig_' + str(opt.backprop_eig)

writer = SummaryWriter(summary_name)


if opt.dataset == 'interiornet':
	train_list_path = root + '/phoenix/S6/wx97/interiornet/train_normal_list.txt'
	eval_list_path = root + '/phoenix/S6/wx97/interiornet/val_normal_list.txt'

	train_num_threads = 3
	train_data_loader = CreateInteriorNetryDataLoader(opt, train_list_path, 
													  True, TRAIN_BATCH_SIZE, 
													  train_num_threads)
	train_dataset = train_data_loader.load_data()
	train_data_size = len(train_data_loader)
	print('========================= InteriorNet training #images = %d ========='%train_data_size)

	iteration_per_epoch = train_data_size//TRAIN_BATCH_SIZE
	eval_num_threads = 3

	test_data_loader = CreateInteriorNetryDataLoader(opt, eval_list_path, 
													False, EVAL_BATCH_SIZE, 
													eval_num_threads)
	test_dataset = test_data_loader.load_data()
	test_data_size = len(test_data_loader)
	print('========================= InteriorNet eval #images = %d ========='%test_data_size)

elif opt.dataset == 'scannet':
	train_list_path = root + '/phoenix/S3/zl548/ScanNet/train_scannet_normal_list.txt'
	eval_list_path = root + '/phoenix/S3/zl548/ScanNet/val_scannet_normal_list.txt'

	train_num_threads = 3
	train_data_loader = CreateScanNetDataLoader(opt, train_list_path, 
												True, TRAIN_BATCH_SIZE, 
												train_num_threads)
	train_dataset = train_data_loader.load_data()
	train_data_size = len(train_data_loader)
	print('========================= ScanNet training #images = %d ========='%train_data_size)

	iteration_per_epoch = train_data_size//TRAIN_BATCH_SIZE
	eval_num_threads = 3

	test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
											   False, EVAL_BATCH_SIZE, 
											   eval_num_threads)
	test_dataset = test_data_loader.load_data()
	test_data_size = len(test_data_loader)
	print('========================= ScanNet eval #images = %d ========='%test_data_size)


iteration_per_epoch_eval = test_data_size//EVAL_BATCH_SIZE


model = create_model(opt, True)
model.switch_to_train()
model.set_writer(writer)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0


def validation_numerical(model, dataset, global_step):
	total_cam_n_error, total_up_n_error, \
	total_rotation_error, total_roll_error, \
	total_pitch_error = 0.,0., 0., 0., 0.
	
	total_cam_u_error, total_up_u_error = 0., 0.

	count = 0.0

	model.switch_to_eval()

	count = 0

	upper_bound = 200

	rad_summary_idx = np.random.randint(0, min(upper_bound, iteration_per_epoch_eval-1))

	for i, data in enumerate(dataset):
		stacked_img = data[0]
		targets = data[1]

		if i == rad_summary_idx:
			write_summary = True
		else:
			write_summary = False

		cam_n_error, cam_u_error, rotation_error, \
		roll_error, pitch_error = model.evaluate_normal_pose(stacked_img, 
															 targets, 
															 global_step, 
															 write_summary)

		total_cam_n_error += cam_n_error
		total_cam_u_error += cam_u_error

		total_rotation_error += rotation_error
		total_roll_error += roll_error
		total_pitch_error += pitch_error
		
		count += stacked_img.size(0)

		avg_rotation_error = float(total_rotation_error)/float(count)
		avg_roll_error = float(total_roll_error)/float(count)
		avg_pitch_error = float(total_pitch_error)/float(count)

		print('iteration_per_epoch_eval ', iteration_per_epoch_eval)
		print('============== avg_rotation_error: %d %f'%(i, avg_rotation_error))
		print('============== avg_roll_error: %d %f'%(i, avg_roll_error))
		print('============== avg_pitch_error: %d %f'%(i, avg_pitch_error))

		if i > upper_bound:
			break

	avg_cam_n_error = float(total_cam_n_error)/float(count)
	avg_cam_u_error = float(total_cam_u_error)/float(count)

	avg_rotation_error = float(total_rotation_error)/float(count)
	avg_roll_error = float(total_roll_error)/float(count)
	avg_pitch_error = float(total_pitch_error)/float(count)

	print('iteration_per_epoch_eval ', iteration_per_epoch_eval)

	print('============== avg_cam_n_error: %d %f'%(i, avg_cam_n_error))
	print('============== avg_cam_u_error: %d %f'%(i, avg_cam_u_error))


	print('============== avg_rotation_error: %d %f'%(i, avg_rotation_error))
	print('============== avg_roll_error: %d %f'%(i, avg_roll_error))
	print('============== avg_pitch_error: %d %f'%(i, avg_pitch_error))

	model.writer.add_scalar('Eval/avg_cam_n_error', 
							avg_cam_n_error, 
							global_step)
	model.writer.add_scalar('Eval/avg_cam_u_error', 
							avg_cam_u_error, 
							global_step)

	model.writer.add_scalar('Eval/avg_rotation_error', 
							avg_rotation_error, 
							global_step)
	model.writer.add_scalar('Eval/avg_roll_error', 
							avg_roll_error, 
							global_step)
	model.writer.add_scalar('Eval/avg_pitch_error', 
							avg_pitch_error, 
							global_step)

	model.switch_to_train()

	return avg_cam_n_error, avg_cam_u_error, \
			avg_rotation_error, avg_roll_error, \
			avg_pitch_error


best_n_error, best_cam_n_error, best_up_n_error, best_cam_u_error, best_up_u_error = 10000, 10000, 10000, 10000, 10000
best_rotation_error, best_roll_error, best_pitch_error = 10000, 10000, 10000
eval_interval = iteration_per_epoch//6
best_epoch = 0


if opt.dataset == 'scannet':
	stop_epoch = 15
else:
	stop_epoch = 20


for epoch in range(0, 25):

	model.update_learning_rate()

	for i, data in enumerate(train_dataset):
		print('%s: epoch %d, iteration %d best_cam_n_error %f best_cam_u_error %f num_iterations %d best_epoch %d'\
			 %(opt.dataset, epoch, i, best_cam_n_error, best_cam_u_error, iteration_per_epoch, best_epoch))
		print('%s-mode-%s-lr-%f-w_pose-%f-w_grad-%f' \
			 % (opt.log_comment, opt.mode, opt.lr, opt.w_pose, opt.w_grad) )
		print('best_rotation_error %f best_roll_error %f best_pitch_error %f' \
			 %(best_rotation_error, best_roll_error, best_pitch_error))
		
		global_step = global_step + 1
		print('global_step', global_step)
		stacked_img = data[0]
		targets = data[1]

		model.set_input(stacked_img, targets)
		model.optimize_parameters(global_step)

		if global_step%eval_interval == 0:

			avg_cam_n_error, avg_cam_u_error, \
			avg_rotation_error, avg_roll_error, \
			avg_pitch_error = validation_numerical(model, test_dataset, global_step)

			if avg_rotation_error < best_rotation_error:
				best_cam_n_error = avg_cam_n_error
				best_cam_u_error = avg_cam_u_error

				best_rotation_error = avg_rotation_error
				best_roll_error = avg_roll_error
				best_pitch_error = avg_pitch_error

				best_epoch = epoch
				model.save('best_%s_%s_mode_'%(opt.dataset, opt.log_comment) \
					+ opt.mode + '_lr_' + str(opt.lr) + '_w_svd_' + str(opt.w_svd) + \
					'_w_grad_' + str(opt.w_grad) + \
					'_backprop_eig_' + str(opt.backprop_eig))

				if epoch >= stop_epoch:
					print('we are done, stop training !!!')
					sys.exit()
