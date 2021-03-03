from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import sys

imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def normalize_normal(normals):
	normals_norm = torch.sqrt( torch.sum(torch.pow(normals , 2) , 1) ).unsqueeze(1).repeat(1,3,1,1) + 1e-8
	return torch.div(normals , normals_norm)

def normalize_coords(coords):
	coord_n = coords[:, 0:3, :, :]
	coord_u = coords[:, 3:6, :, :]

	return torch.cat((normalize_normal(coord_n), normalize_normal(coord_u)), 1)


class conv(nn.Module):
	def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
		super(conv, self).__init__()
		self.kernel_size = kernel_size
		self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
		self.normalize = nn.BatchNorm2d(num_out_layers)

	def forward(self, x):
		p = int(np.floor((self.kernel_size-1)/2))
		p2d = (p, p, p, p)
		x = self.conv_base(F.pad(x, p2d))
		x = self.normalize(x)
		return F.elu(x, inplace=True)


class convblock(nn.Module):
	def __init__(self, num_in_layers, num_out_layers, kernel_size):
		super(convblock, self).__init__()
		self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
		self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

	def forward(self, x):
		x = self.conv1(x)
		return self.conv2(x)


class maxpool(nn.Module):
	def __init__(self, kernel_size):
		super(maxpool, self).__init__()
		self.kernel_size = kernel_size

	def forward(self, x):
		p = int(np.floor((self.kernel_size-1) / 2))
		p2d = (p, p, p, p)
		return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
	def __init__(self, num_in_layers, num_out_layers, stride):
		super(resconv, self).__init__()
		self.num_out_layers = num_out_layers
		self.stride = stride
		self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
		self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
		self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
		self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
		self.normalize = nn.BatchNorm2d(4*num_out_layers)

	def forward(self, x):
		# do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
		do_proj = True
		shortcut = []
		x_out = self.conv1(x)
		x_out = self.conv2(x_out)
		x_out = self.conv3(x_out)
		if do_proj:
			shortcut = self.conv4(x)
		else:
			shortcut = x
		return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
	# for resnet18
	def __init__(self, num_in_layers, num_out_layers, stride):
		super(resconv_basic, self).__init__()
		self.num_out_layers = num_out_layers
		self.stride = stride
		self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
		self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
		self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
		self.normalize = nn.BatchNorm2d(num_out_layers)

	def forward(self, x):
		#         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
		do_proj = True
		shortcut = []
		x_out = self.conv1(x)
		x_out = self.conv2(x_out)
		if do_proj:
			shortcut = self.conv3(x)
		else:
			shortcut = x
		return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
	layers = []
	layers.append(resconv(num_in_layers, num_out_layers, stride))
	for i in range(1, num_blocks - 1):
		layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
	layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
	return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
	layers = []
	layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
	for i in range(1, num_blocks):
		layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
	return nn.Sequential(*layers)


class upconv(nn.Module):
	def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
		super(upconv, self).__init__()
		self.scale = scale
		self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

	def forward(self, x):
		x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
		return self.conv1(x)


class downsampled_get_normal(nn.Module):
	def __init__(self, num_in_layers):
		super(downsampled_get_normal, self).__init__()
		self.conv1 = nn.Conv2d(num_in_layers, 3, kernel_size=3, stride=2)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		p = 1
		p2d = (p, p, p, p)
		x_out = self.conv1(F.pad(x, p2d))
		
		return x_out

class get_confidence(nn.Module):
	def __init__(self, num_in_layers, num_out_layers=1):
		super(get_confidence, self).__init__()
		self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3, stride=1)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		p = 1
		p2d = (p, p, p, p)
		x_out = self.conv1(F.pad(x, p2d))

		return self.sigmoid(x_out)

class get_normal(nn.Module):
	def __init__(self, num_in_layers, num_out_layers=3):
		super(get_normal, self).__init__()
		self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3, stride=1)
		# self.normalize = nn.BatchNorm2d(2)
		# self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		p = 1
		p2d = (p, p, p, p)
		x_out = self.conv1(F.pad(x, p2d))



		return x_out

class Resnet50_md(nn.Module):
	def __init__(self, num_in_layers):
		super(Resnet50_md, self).__init__()
		# encoder
		self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
		self.pool1 = maxpool(3)  # H/4  -   64D
		self.conv2 = resblock(64, 64, 3, 2)  # H/8  -  256D
		self.conv3 = resblock(256, 128, 4, 2)  # H/16 -  512D
		self.conv4 = resblock(512, 256, 6, 2)  # H/32 - 1024D
		self.conv5 = resblock(1024, 512, 3, 2)  # H/64 - 2048D

		# decoder
		self.upconv6 = upconv(2048, 512, 3, 2)
		self.iconv6 = conv(1024 + 512, 512, 3, 1)

		self.upconv5 = upconv(512, 256, 3, 2)
		self.iconv5 = conv(512+256, 256, 3, 1)

		self.upconv4 = upconv(256, 128, 3, 2)
		self.iconv4 = conv(256+128, 128, 3, 1)
		self.normal4_layer = get_normal(128)

		self.upconv3 = upconv(128, 64, 3, 2)
		self.iconv3 = conv(64+64+1, 64, 3, 1)
		self.normal3_layer = get_normal(64)

		self.upconv2 = upconv(64, 32, 3, 2)
		self.iconv2 = conv(32+64+1, 32, 3, 1)
		self.normal2_layer = get_normal(32)

		self.upconv1 = upconv(32, 16, 3, 2)
		self.iconv1 = conv(16+1, 16, 3, 1)
		self.normal1_layer = get_normal(16)

		# for m in self.modules():
			# if isinstance(m, nn.Conv2d):
				# nn.init.xavier_uniform_(m.weight)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# nn.init.xavier_uniform_(m.weight)
				# m.weight.data.normal_(0.0, 0.02)
				m.bias.data = torch.zeros(m.bias.data.size())


	def forward(self, x):
		# encoder
		x1 = self.conv1(x)
		x_pool1 = self.pool1(x1)
		x2 = self.conv2(x_pool1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)

		# skips
		skip1 = x1
		skip2 = x_pool1
		skip3 = x2
		skip4 = x3
		skip5 = x4

		# decoder
		upconv6 = self.upconv6(x5)
		concat6 = torch.cat((upconv6, skip5), 1)
		iconv6 = self.iconv6(concat6)

		upconv5 = self.upconv5(iconv6)
		concat5 = torch.cat((upconv5, skip4), 1)
		iconv5 = self.iconv5(concat5)

		upconv4 = self.upconv4(iconv5)
		concat4 = torch.cat((upconv4, skip3), 1)
		iconv4 = self.iconv4(concat4)
		self.normal4 = self.normal4_layer(iconv4)
		self.unormal4 = nn.functional.interpolate(self.normal4, scale_factor=2, mode='bilinear', align_corners=True)

		upconv3 = self.upconv3(iconv4)
		concat3 = torch.cat((upconv3, skip2, self.unormal4), 1)
		iconv3 = self.iconv3(concat3)
		self.normal3 = self.normal3_layer(iconv3)
		self.unormal3 = nn.functional.interpolate(self.normal3, scale_factor=2, mode='bilinear', align_corners=True)

		upconv2 = self.upconv2(iconv3)
		concat2 = torch.cat((upconv2, skip1, self.unormal3), 1)
		iconv2 = self.iconv2(concat2)
		self.normal2 = self.normal2_layer(iconv2)
		self.unormal2 = nn.functional.interpolate(self.normal2, scale_factor=2, mode='bilinear', align_corners=True)

		upconv1 = self.upconv1(iconv2)
		concat1 = torch.cat((upconv1, self.unormal2), 1)
		iconv1 = self.iconv1(concat1)
		self.normal1 = self.normal1_layer(iconv1)
		return [self.normal1, self.normal2, self.normal3, self.normal4]


class Resnet18_md(nn.Module):
	def __init__(self, num_in_layers):
		super(Resnet18_md, self).__init__()
		# encoder
		self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
		self.pool1 = maxpool(3)  # H/4  -   64D
		self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
		self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
		self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
		self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

		# decoder
		self.upconv6 = upconv(512, 512, 3, 2)
		self.iconv6 = conv(256+512, 512, 3, 1)

		self.upconv5 = upconv(512, 256, 3, 2)
		self.iconv5 = conv(128+256, 256, 3, 1)

		self.upconv4 = upconv(256, 128, 3, 2)
		self.iconv4 = conv(64+128, 128, 3, 1)
		self.normal4_layer = get_normal(128)

		self.upconv3 = upconv(128, 64, 3, 2)
		self.iconv3 = conv(64+64 + 2, 64, 3, 1)
		self.normal3_layer = get_normal(64)

		self.upconv2 = upconv(64, 32, 3, 2)
		self.iconv2 = conv(64+32 + 2, 32, 3, 1)
		self.normal2_layer = get_normal(32)

		self.upconv1 = upconv(32, 16, 3, 2)
		self.iconv1 = conv(16+2, 16, 3, 1)
		self.normal1_layer = get_normal(16)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):
		# encoder
		x1 = self.conv1(x)
		x_pool1 = self.pool1(x1)
		x2 = self.conv2(x_pool1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)

		# skips
		skip1 = x1
		skip2 = x_pool1
		skip3 = x2
		skip4 = x3
		skip5 = x4

		# decoder
		upconv6 = self.upconv6(x5)
		concat6 = torch.cat((upconv6, skip5), 1)
		iconv6 = self.iconv6(concat6)

		upconv5 = self.upconv5(iconv6)
		concat5 = torch.cat((upconv5, skip4), 1)
		iconv5 = self.iconv5(concat5)

		upconv4 = self.upconv4(iconv5)
		concat4 = torch.cat((upconv4, skip3), 1)
		iconv4 = self.iconv4(concat4)
		self.normal4 = self.normal4_layer(iconv4)
		self.unormal4 = nn.functional.interpolate(self.normal4, scale_factor=2, mode='bilinear', align_corners=True)

		upconv3 = self.upconv3(iconv4)
		concat3 = torch.cat((upconv3, skip2, self.unormal4), 1)
		iconv3 = self.iconv3(concat3)
		self.normal3 = self.normal3_layer(iconv3)
		self.unormal3 = nn.functional.interpolate(self.normal3, scale_factor=2, mode='bilinear', align_corners=True)

		upconv2 = self.upconv2(iconv3)
		concat2 = torch.cat((upconv2, skip1, self.unormal3), 1)
		iconv2 = self.iconv2(concat2)
		self.normal2 = self.normal2_layer(iconv2)
		self.unormal2 = nn.functional.interpolate(self.normal2, scale_factor=2, mode='bilinear', align_corners=True)

		upconv1 = self.upconv1(iconv2)
		concat1 = torch.cat((upconv1, self.unormal2), 1)
		iconv1 = self.iconv1(concat1)
		self.normal1 = self.normal1_layer(iconv1)
		return self.normal1, self.normal2, self.normal3, self.normal4


def class_for_name(module_name, class_name):
	# load the module, will raise ImportError if module cannot be loaded
	m = importlib.import_module(module_name)
	# get the class, will raise AttributeError if class cannot be found
	return getattr(m, class_name)



class ResnetModel3HeadsSplitNormalizationTightCoupled(nn.Module):
	def __init__(self, num_in_layers, encoder, pretrained):
		super(ResnetModel3HeadsSplitNormalizationTightCoupled, self).__init__()
		self.pretrained = pretrained
		assert encoder in ['resnet18', 'resnet34', 'resnet50',\
						   'resnet101', 'resnet152'],\
						   "Incorrect encoder type"
		if encoder in ['resnet18', 'resnet34']:
			filters = [64, 128, 256, 512]
		else:
			filters = [256, 512, 1024, 2048]
		resnet = class_for_name("torchvision.models", encoder)\
								(pretrained=pretrained)
		if num_in_layers != 3:  # Number of input channels
			self.firstconv = nn.Conv2d(num_in_layers, 64,
							  kernel_size=(7, 7), stride=(2, 2),
							  padding=(3, 3), bias=False)
		else:
			self.firstconv = resnet.conv1 # H/2
		self.firstbn = resnet.bn1
		self.firstrelu = resnet.relu
		self.firstmaxpool = resnet.maxpool # H/4

		# encoder
		self.encoder1 = resnet.layer1 # H/4
		self.encoder2 = resnet.layer2 # H/8
		self.encoder3 = resnet.layer3 # H/16
		self.encoder4 = resnet.layer4 # H/32

		# decoder camera 
		self.upconv6_cam = upconv(filters[3], 512, 3, 2)
		self.iconv6_cam = conv(filters[2] + 512, 512, 3, 1)

		self.upconv5_cam = upconv(512, 256, 3, 2)
		self.iconv5_cam = conv(filters[1] + 256, 256, 3, 1)

		self.upconv4_cam = upconv(256, 128, 3, 2)
		self.iconv4_cam = conv(filters[0] + 128, 128, 3, 1)

		self.upconv3_cam = upconv(128, 64, 3, 1) #
		self.iconv3_cam = conv(64 + 64, 64, 3, 1)

		self.upconv2_cam = upconv(64, 32, 3, 2)
		self.iconv2_cam = conv(64 + 32, 32, 3, 1)

		self.upconv1_cam = upconv(32, 16, 3, 2)

		self.iconv1_cam = conv(16, 16, 3, 1)
		self.nu_layer_cam = get_normal(16, 9)

		# decoder upright 
		self.upconv6_upright = upconv(filters[3], 512, 3, 2)
		self.iconv6_upright = conv(filters[2] + 512, 512, 3, 1)

		self.upconv5_upright = upconv(512, 256, 3, 2)
		self.iconv5_upright = conv(filters[1] + 256, 256, 3, 1)

		self.upconv4_upright = upconv(256, 128, 3, 2)
		self.iconv4_upright = conv(filters[0] + 128, 128, 3, 1)

		self.upconv3_upright = upconv(128, 64, 3, 1) #
		self.iconv3_upright = conv(64 + 64, 64, 3, 1)

		self.upconv2_upright = upconv(64, 32, 3, 2)
		self.iconv2_upright = conv(64 + 32, 32, 3, 1)

		self.upconv1_upright = upconv(32, 16, 3, 2)

		self.iconv1_upright = conv(16, 16, 3, 1)
		self.nu_layer_upright = get_normal(16, 3)

		# decoder confidence 
		self.upconv6_confidence = upconv(filters[3], 512, 3, 2)
		self.iconv6_confidence = conv(filters[2] + 512, 512, 3, 1)

		self.upconv5_confidence = upconv(512, 256, 3, 2)
		self.iconv5_confidence = conv(filters[1] + 256, 256, 3, 1)

		self.upconv4_confidence = upconv(256, 128, 3, 2)
		self.iconv4_confidence = conv(filters[0] + 128, 128, 3, 1)

		self.upconv3_confidence = upconv(128, 64, 3, 1) #
		self.iconv3_confidence = conv(64 + 64, 64, 3, 1)

		self.upconv2_confidence = upconv(64, 32, 3, 2)
		self.iconv2_confidence = conv(64 + 32, 32, 3, 1)

		self.upconv1_confidence = upconv(32, 16, 3, 2)

		self.iconv1_confidence = conv(16, 16, 3, 1)
		self.nu_layer_confidence = get_confidence(16, num_out_layers=3)

		# self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		# self.fc_1 = nn.Linear(512 * 4, 1)

	def forward(self, x):

		if self.pretrained:
			x_normalized_r = (x[:, 0, :, :] - imagenet_stats['mean'][0])/imagenet_stats['std'][0]
			x_normalized_g = (x[:, 1, :, :] - imagenet_stats['mean'][1])/imagenet_stats['std'][1]
			x_normalized_b = (x[:, 2, :, :] - imagenet_stats['mean'][2])/imagenet_stats['std'][2]
			x_normalized = torch.cat((x_normalized_r.unsqueeze(1), x_normalized_g.unsqueeze(1), x_normalized_b.unsqueeze(1)), 1)
		else:
			x_normalized = x

		# encoder
		x_first_conv = self.firstconv(x_normalized)
		x_2 = self.firstbn(x_first_conv)
		x_2 = self.firstrelu(x_2)
		x_pool1 = self.firstmaxpool(x_2)

		x1 = self.encoder1(x_pool1)
		x2 = self.encoder2(x1)
		x3 = self.encoder3(x2)
		x4 = self.encoder4(x3)

		# x5 = self.avgpool(x4)
		# x5 = x5.view(x5.size(0), -1)
		# self.pred_lambda = self.fc_1(x5)

		# skips
		skip1 = x_first_conv
		skip2 = x_pool1
		skip3 = x1
		skip4 = x2
		skip5 = x3  

		# decoder camera 
		upconv6_cam = self.upconv6_cam(x4)
		concat6_cam = torch.cat((upconv6_cam, skip5), 1)
		iconv6_cam = self.iconv6_cam(concat6_cam)

		upconv5_cam = self.upconv5_cam(iconv6_cam)
		concat5_cam = torch.cat((upconv5_cam, skip4), 1)
		iconv5_cam = self.iconv5_cam(concat5_cam)

		upconv4_cam = self.upconv4_cam(iconv5_cam)
		concat4_cam = torch.cat((upconv4_cam, skip3), 1)
		iconv4_cam = self.iconv4_cam(concat4_cam)

		upconv3_cam = self.upconv3_cam(iconv4_cam)
		concat3_cam = torch.cat((upconv3_cam, skip2), 1)
		iconv3_cam = self.iconv3_cam(concat3_cam)

		upconv2_cam = self.upconv2_cam(iconv3_cam)
		concat2_cam = torch.cat((upconv2_cam, skip1), 1)
		iconv2_cam = self.iconv2_cam(concat2_cam)

		upconv1_cam = self.upconv1_cam(iconv2_cam)
		concat1_cam = upconv1_cam 

		self.pred_cam_geo = self.nu_layer_cam(self.iconv1_cam(concat1_cam)) #torch.cat((pred_cam_n, pred_cam_u), 1)

		# decoder upright 
		upconv6_upright = self.upconv6_upright(x4)
		concat6_upright = torch.cat((upconv6_upright, skip5), 1)
		iconv6_upright = self.iconv6_upright(concat6_upright)

		upconv5_upright = self.upconv5_upright(iconv6_upright)
		concat5_upright = torch.cat((upconv5_upright, skip4), 1)
		iconv5_upright = self.iconv5_upright(concat5_upright)

		upconv4_upright = self.upconv4_upright(iconv5_upright)
		concat4_upright = torch.cat((upconv4_upright, skip3), 1)
		iconv4_upright = self.iconv4_upright(concat4_upright)

		upconv3_upright = self.upconv3_upright(iconv4_upright)
		concat3_upright = torch.cat((upconv3_upright, skip2), 1)
		iconv3_upright = self.iconv3_upright(concat3_upright)

		upconv2_upright = self.upconv2_upright(iconv3_upright)
		concat2_upright = torch.cat((upconv2_upright, skip1), 1)
		iconv2_upright = self.iconv2_upright(concat2_upright)

		upconv1_upright = self.upconv1_upright(iconv2_upright)
		concat1_upright = upconv1_upright

		self.pred_upright_geo = self.nu_layer_upright(self.iconv1_upright(concat1_upright))

		# decoder confidence 
		upconv6_confidence = self.upconv6_confidence(x4)
		concat6_confidence = torch.cat((upconv6_confidence, skip5), 1)
		iconv6_confidence = self.iconv6_confidence(concat6_confidence)

		upconv5_confidence = self.upconv5_confidence(iconv6_confidence)
		concat5_confidence = torch.cat((upconv5_confidence, skip4), 1)
		iconv5_confidence = self.iconv5_confidence(concat5_confidence)

		upconv4_confidence = self.upconv4_confidence(iconv5_confidence)
		concat4_confidence = torch.cat((upconv4_confidence, skip3), 1)
		iconv4_confidence = self.iconv4_confidence(concat4_confidence)

		upconv3_confidence = self.upconv3_confidence(iconv4_confidence)
		concat3_confidence = torch.cat((upconv3_confidence, skip2), 1)
		iconv3_confidence = self.iconv3_confidence(concat3_confidence)

		upconv2_confidence = self.upconv2_confidence(iconv3_confidence)
		concat2_confidence = torch.cat((upconv2_confidence, skip1), 1)
		iconv2_confidence = self.iconv2_confidence(concat2_confidence)

		upconv1_confidence = self.upconv1_confidence(iconv2_confidence)

		concat1_confidence = upconv1_confidence #torch.cat((upconv1_confidence, cam_nu_unit, up_nu_unit), 1)#upconv1_confidence

		weights_geo = self.nu_layer_confidence(self.iconv1_confidence(concat1_confidence))

		mean_weights = torch.mean(weights_geo.view(weights_geo.size(0), weights_geo.size(1) * weights_geo.size(2) * weights_geo.size(3)), dim=-1, keepdim=True)#.unsqueeze(-1).repeat(1,1, confidence_2.size(2), confidence_2.size(3))
		mean_weights = mean_weights.unsqueeze(-1).unsqueeze(-1).repeat(1, weights_geo.size(1), weights_geo.size(2), weights_geo.size(3))
		self.pred_weights_geo = weights_geo/mean_weights

		return self.pred_cam_geo, self.pred_upright_geo, self.pred_weights_geo#, self.pred_lambda 

