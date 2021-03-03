from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.test_options import TestOptions 
import sys
from data.data_loader import *
from models.models import create_model
import random
from tensorboardX import SummaryWriter


EVAL_BATCH_SIZE = 8
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = '/'


if opt.dataset == 'interiornet':
	eval_list_path = root + '/phoenix/S6/wx97/interiornet/test_normal_list.txt'
	eval_num_threads = 3
	test_data_loader = CreateInteriorNetryDataLoader(opt, eval_list_path, 
													False, EVAL_BATCH_SIZE, 
													eval_num_threads)
	test_dataset = test_data_loader.load_data()
	test_data_size = len(test_data_loader)
	print('========================= InteriorNet Test #images = %d ========='%test_data_size)


elif opt.dataset == 'scannet':
	eval_list_path = root + '/phoenix/S3/zl548/ScanNet/test_scannet_normal_list.txt'
	eval_num_threads = 2
	test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
													False, EVAL_BATCH_SIZE, 
													eval_num_threads)
	test_dataset = test_data_loader.load_data()
	test_data_size = len(test_data_loader)
	print('========================= ScanNet eval #images = %d ========='%test_data_size)


elif opt.dataset == 'sun360':
	eval_list_path = root + '/phoenix/S6/wx97/sun360/sun360_indoor_test_uniform_list.txt'
	# iteration_per_epoch = train_data_size//TRAIN_BATCH_SIZE
	eval_num_threads = 2

	test_data_loader = CreateSUN360DataLoader(opt, eval_list_path, False, EVAL_BATCH_SIZE, eval_num_threads)
	test_dataset = test_data_loader.load_data()
	test_data_size = len(test_data_loader)
	print('========================= sun360 eval #images = %d ========='%test_data_size)


else:
	print('INPUT DATASET DOES NOT EXIST!!!')
	sys.exit()

model = create_model(opt, _isTrain=False)
model.switch_to_train()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0


def test_numerical(model, dataset, global_step):
	rot_e_list = []
	roll_e_list = []
	pitch_e_list = []

	count = 0.0

	model.switch_to_eval()

	count = 0

	for i, data in enumerate(dataset):
		stacked_img = data[0]
		targets = data[1]

		rotation_error, roll_error, pitch_error = model.test_angle_error(stacked_img, targets)

		rot_e_list = rot_e_list + rotation_error	
		roll_e_list = roll_e_list + roll_error
		pitch_e_list = pitch_e_list + pitch_error

		rot_e_arr = np.array(rot_e_list)
		roll_e_arr = np.array(roll_e_list)
		pitch_e_arr = np.array(pitch_e_list)

		mean_rot_e = np.mean(rot_e_arr)
		# median_rot_e = np.median(rot_e_arr)
		std_rot_e = np.std(rot_e_arr)

		mean_roll_e = np.mean(roll_e_arr)
		# median_roll_e = np.median(roll_e_arr)
		std_roll_e = np.std(roll_e_arr)

		mean_pitch_e = np.mean(pitch_e_arr)
		# median_pitch_e = np.median(pitch_e_arr)
		std_pitch_e = np.std(pitch_e_arr)

		print(i)

	rot_e_arr = np.array(rot_e_list)
	roll_e_arr = np.array(roll_e_list)
	pitch_e_arr = np.array(pitch_e_list)

	mean_rot_e = np.mean(rot_e_arr)
	median_rot_e = np.median(rot_e_arr)
	std_rot_e = np.std(rot_e_arr)

	mean_roll_e = np.mean(roll_e_arr)
	median_roll_e = np.median(roll_e_arr)
	std_roll_e = np.std(roll_e_arr)

	mean_pitch_e = np.mean(pitch_e_arr)
	median_pitch_e = np.median(pitch_e_arr)
	std_pitch_e = np.std(pitch_e_arr)

	print('======================= FINAL STATISCIS ==========================')
	print('mean_rot_e ', mean_rot_e)
	print('median_rot_e ', median_rot_e)
	print('std_rot_e ', std_rot_e)

	print('mean_roll_e ', mean_roll_e)
	print('median_roll_e ', median_roll_e)
	print('std_roll_e ', std_roll_e)

	print('mean_pitch_e ', mean_pitch_e)
	print('median_pitch_e ', median_pitch_e)	
	print('std_pitch_e ', std_pitch_e)


test_numerical(model, test_dataset, global_step)
