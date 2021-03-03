from __future__ import division 
import numpy as np
import torch
import os

from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
import sys
import os.path
import cv2
import torchvision.utils as vutils
import torch.nn as nn

import models.models_resnet as models_resnet


EPSILON = 1e-8


class UprightNet(BaseModel):
    def name(self):
        return 'UprightNet'

    def __init__(self, opt, _isTrain):
        BaseModel.initialize(self, opt)

        self.mode = opt.mode
        self.num_input = 3#opt.input_nc

        if self.mode == 'ResNet':
            new_model = models_resnet.ResnetModel3HeadsSplitNormalizationTightCoupled(num_in_layers=3, 
                                                                                    encoder='resnet50', 
                                                                                    pretrained=True)
        else:
            print('ONLY SUPPORT Ours_Bilinear')
            sys.exit()

        if not _isTrain:
            if opt.dataset == 'interiornet':            
                model_name = '_best_interiornet_ry_exp_upright_9_sphere_ls_mode_ResnetModel3HeadsSplitNormalizationTightCoupled_lr_0.0004_w_svd_2.0_w_grad_0.25_backprop_eig_1'
            else:
                model_name = '_best_scannet_exp_upright_9_sphere_ls_mode_ResnetModel3HeadsSplitNormalizationTightCoupled_lr_0.0004_w_svd_0.5_w_grad_0.25_backprop_eig_1'

            model_parameters = self.load_network(new_model, 'G', model_name)
            new_model.load_state_dict(model_parameters)

        new_model = torch.nn.parallel.DataParallel(new_model.cuda(), 
                                                  device_ids = [0])
        self.netG = new_model
        self.criterion_joint = networks.JointLoss(opt) 

        if _isTrain:      
            self.old_lr = opt.lr
            self.netG.train()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer_G, opt)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_writer(self, writer):
        self.writer = writer

    def set_input(self, stack_imgs, targets):
        self.input = stack_imgs
        self.targets = targets

    def forward(self):
        self.input_images = Variable(self.input.cuda(), requires_grad = False)

        self.pred_cam_geo, self.pred_up_geo, self.pred_weights = self.netG.forward(self.input_images)

        self.pred_cam_geo_unit = self.criterion_joint.normalize_coords(self.pred_cam_geo)
        self.pred_up_geo_unit = self.criterion_joint.normalize_normal(self.pred_up_geo)


    def get_image_paths(self):
        return self.image_paths


    def write_summary_train(self, mode_name, input_images,
                            pred_cam_geo_unit, 
                            pred_up_geo_unit,
                            pred_weights,
                            cam_n_term, upright_n_term, 
                            pose_term,
                            targets, n_iter):

        pred_cam_geo = torch.cat((pred_cam_geo_unit[:, 0:3, :, :], 
                                  pred_cam_geo_unit[:, 3:6, :, :], 
                                  pred_cam_geo_unit[:, 6:9, :, :]), 2)

        pred_cam_geo_rgb = (pred_cam_geo + 1.)/2.
        pred_up_geo_rgb = (pred_up_geo_unit + 1.)/2.

        gt_cam_geo = torch.cat((targets['cam_geo'][:, 0:3, :, :], 
                                targets['cam_geo'][:, 3:6, :, :], 
                                targets['cam_geo'][:, 6:9, :, :]), 2) 

        gt_cam_geo_rgb = (gt_cam_geo + 1.0)/2.0
        gt_up_geo_rgb = (targets['upright_geo'] + 1.0)/2.0

        pred_w_n_rgb = pred_weights[:, 0:1, :, :].repeat(1,3,1,1)
        pred_w_u_rgb = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)
        pred_w_t_rgb = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)

        pred_weights_rgb = torch.cat( (pred_w_n_rgb, 
                                       pred_w_u_rgb, 
                                       pred_w_t_rgb), dim=2)

        num_write = 6

        self.writer.add_scalar(mode_name + '/cam_n_term', cam_n_term, n_iter)
        self.writer.add_scalar(mode_name + '/upright_n_term', upright_n_term, n_iter)
        self.writer.add_scalar(mode_name + '/pose_term', pose_term, n_iter)

        self.writer.add_image(mode_name + '/img', 
                              vutils.make_grid(input_images.data[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)

        self.writer.add_image(mode_name + '/gt_cam_geo', 
                              vutils.make_grid(gt_cam_geo_rgb[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_cam_geo', 
                              vutils.make_grid(pred_cam_geo_rgb[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)

        self.writer.add_image(mode_name + '/gt_up_geo', 
                              vutils.make_grid(gt_up_geo_rgb[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_up_geo', 
                              vutils.make_grid(pred_up_geo_rgb[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)

        self.writer.add_image(mode_name + '/pred_weights_geo', 
                              vutils.make_grid(pred_weights_rgb.data[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_w_n', 
                              vutils.make_grid(pred_w_n_rgb.data[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_w_u', 
                              vutils.make_grid(pred_w_u_rgb.data[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_w_t', 
                              vutils.make_grid(pred_w_t_rgb.data[:num_write,:,:,:].cpu(), 
                                            normalize=True), n_iter)


    def write_summary_val(self, mode_name, input_images,
                          pred_cam_geo_unit, 
                          pred_up_geo_unit,
                          pred_weights,
                          targets, n_iter):

        pred_cam_geo = torch.cat((pred_cam_geo_unit[:, 0:3, :, :], 
                                  pred_cam_geo_unit[:, 3:6, :, :], 
                                  pred_cam_geo_unit[:, 6:9, :, :]), 2)
        pred_cam_geo_rgb = (pred_cam_geo + 1.)/2.

        # pred_up_geo = torch.cat((pred_up_geo_unit[:, 0:1, :, :].repeat(1,3,1,1), pred_up_geo_unit[:, 1:2, :, :].repeat(1,3,1,1)), 2)
        pred_up_geo_rgb = (pred_up_geo_unit + 1.)/2.

        gt_cam_geo = torch.cat((targets['cam_geo'][:, 0:3, :, :], 
                                targets['cam_geo'][:, 3:6, :, :], 
                                targets['cam_geo'][:, 6:9, :, :]), 2) 
        gt_cam_geo_rgb = (gt_cam_geo + 1.0)/2.0

        gt_up_geo_rgb = (targets['upright_geo'] + 1.0)/2.0

        pred_w_n_rgb = pred_weights[:, 0:1, :, :].repeat(1,3,1,1)
        pred_w_u_rgb = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)
        pred_w_t_rgb = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)

        pred_weights_rgb = torch.cat( (pred_w_n_rgb, 
                                       pred_w_u_rgb, 
                                       pred_w_t_rgb), dim=2)

        num_write = 6
        self.writer.add_image(mode_name + '/img', 
                              vutils.make_grid(input_images.data[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)

        self.writer.add_image(mode_name + '/gt_cam_geo', 
                              vutils.make_grid(gt_cam_geo_rgb[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_cam_geo', 
                              vutils.make_grid(pred_cam_geo_rgb[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)

        self.writer.add_image(mode_name + '/gt_up_geo', 
                              vutils.make_grid(gt_up_geo_rgb[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_up_geo', 
                              vutils.make_grid(pred_up_geo_rgb[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)

        self.writer.add_image(mode_name + '/pred_weights_geo', 
                              vutils.make_grid(pred_weights_rgb.data[:num_write,:,:,:].cpu(), 
                                               normalize=True), n_iter)

        self.writer.add_image(mode_name + '/pred_w_n', 
                              vutils.make_grid(pred_w_n_rgb.data[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_w_u', 
                              vutils.make_grid(pred_w_u_rgb.data[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)
        self.writer.add_image(mode_name + '/pred_w_t', 
                              vutils.make_grid(pred_w_t_rgb.data[:num_write,:,:,:].cpu(), 
                                                normalize=True), n_iter)


    def backward_G(self, n_iter):
        # Combined loss
        self.loss_joint, cam_n_term, \
        upright_n_term, pose_term = self.criterion_joint(self.input_images, 
                                                         self.pred_cam_geo_unit, 
                                                         self.pred_up_geo_unit,
                                                         self.pred_weights,
                                                         self.targets)
        print("Train loss is %f "%self.loss_joint)

        # # add to tensorboard
        if n_iter % 500 == 0:

            self.write_summary_train('Train', self.input_images,
                                     self.pred_cam_geo_unit, 
                                     self.pred_up_geo_unit,
                                     self.pred_weights,
                                     cam_n_term, upright_n_term, 
                                     pose_term,
                                     self.targets, n_iter)

        self.loss_joint_var = self.criterion_joint.get_loss_var()
        self.loss_joint_var.backward()

    def optimize_parameters(self, n_iter):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(n_iter)
        self.optimizer_G.step()

    def evaluate_normal_pose(self, input_, targets, n_iter, write_summary):

        # switch to evaluation mode
        with torch.no_grad():           
            input_imgs = Variable(input_.cuda() , requires_grad = False)

            pred_cam_geo, pred_up_geo, pred_weights = self.netG.forward(input_imgs)
            # normalize predicted surface nomral
            pred_cam_geo_unit = self.criterion_joint.normalize_coords(pred_cam_geo)
            pred_up_geo_unit = self.criterion_joint.normalize_normal(pred_up_geo)

            gt_cam_geo = targets['cam_geo'].cuda()
            gt_upright_geo = targets['upright_geo'].cuda()
            gt_mask = targets['gt_mask'].cuda()

            cam_n_error = self.criterion_joint.compute_normal_error(pred_cam_geo[:, 0:3, :, :].data, 
                                                                    gt_cam_geo[:, 0:3, :, :], 
                                                                    gt_mask)
            cam_u_error = self.criterion_joint.compute_normal_error(pred_cam_geo[:, 3:6, :, :].data, 
                                                                    gt_cam_geo[:, 3:6, :, :], 
                                                                    gt_mask)

            rotation_error, roll_error, pitch_error = self.criterion_joint.compute_angle_error(pred_cam_geo_unit.data, 
                                                                                               pred_up_geo_unit.data,
                                                                                               pred_weights,
                                                                                               targets)

            if write_summary:
                print('==================== WRITING EVAL SUMMARY ==================')
                self.write_summary_val('Eval', input_imgs,
                                        pred_cam_geo_unit, 
                                        pred_up_geo_unit,
                                        pred_weights,
                                        targets, n_iter)

            return cam_n_error, cam_u_error, rotation_error, roll_error, pitch_error


    def test_angle_error(self, input_, targets):

        # switch to evaluation mode
        with torch.no_grad():           
            input_imgs = Variable(input_.cuda() , requires_grad = False)

            pred_cam_geo, pred_up_geo, pred_weights = self.netG.forward(input_imgs)
            # normalize predicted surface nomral
            pred_cam_geo_unit = self.criterion_joint.normalize_coords(pred_cam_geo)
            pred_up_geo_unit = self.criterion_joint.normalize_normal(pred_up_geo)


            rotation_error, roll_error, \
            pitch_error = self.criterion_joint.compute_angle_error(pred_cam_geo_unit.data, 
                                                                   pred_up_geo_unit.data,
                                                                   pred_weights,
                                                                   targets, stack_error=True)

        return rotation_error, roll_error, pitch_error

    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        print('Current learning rate = %.4f' % lr)






