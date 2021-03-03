from __future__ import division
import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
import sys
from torch.autograd import Function
import math
from scipy import misc 
import scipy
import functools
from torch.nn import init
from torch.optim import lr_scheduler

###############################################################################
# Functions
###############################################################################
VERSION = 4
EPSILON = 1e-8


def decompose_rotation(R):
    roll = math.atan2(R[2,1], R[2,2]);          
    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2])); 
    yaw = math.atan2(R[1,0], R[0,0]);        
    return [roll,pitch,yaw]

def decompose_up_n(up_n):
    pitch = - math.asin(up_n[0])

    sin_roll = up_n[1]/math.cos(pitch)

    roll = math.asin(sin_roll)
    return roll, pitch


def compose_rotation(x, y, z):
    X = np.identity(3); 
    Y = np.identity(3); 
    Z = np.identity(3); 

    X[1,1] = math.cos(x)
    X[1,2] = -math.sin(x)
    X[2,1] = math.sin(x)
    X[2,2] = math.cos(x)

    Y[0,0] = math.cos(y)
    Y[0,2] = math.sin(y)
    Y[2,0] = -math.sin(y)
    Y[2,2] = math.cos(y)

    Z[0,0] = math.cos(z)
    Z[0,1] = -math.sin(z)
    Z[1,0] = math.sin(z)
    Z[1,1] = math.cos(z)

    # R = np.dot(Z, np.dot(Y, X))

    R = np.matmul(np.matmul(Z,Y),X)
    return R

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def print_network(net_):
    num_params = 0
    for param in net_.parameters():
        num_params += param.numel()
    #print(net_)
    #print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class ExtractSMinEigenValue(Function):
    @staticmethod
    def forward(ctx, input):
        D, VL, VR = scipy.linalg.eig(input.detach().cpu().numpy(), left=True, right=True)

        real_idx = np.abs(D.imag) < EPSILON

        vl= VL[:, real_idx]
        vr = VR[:, real_idx]
        lambda_real = D[real_idx].real

        min_idx = np.argmin(lambda_real)
        min_lambda = lambda_real[min_idx] 
        mim_vl = vl[:, min_idx].real
        min_vr = vr[:, min_idx].real

        # normalize vl so that vl'*vr = I
        sum_vlvr = np.dot(mim_vl, min_vr)
        mim_vl = mim_vl/sum_vlvr

        ctx.save_for_backward(input, 
                              torch.as_tensor(mim_vl, 
                                              dtype=input.dtype, 
                                              device=torch.device('cuda')), 
                              torch.as_tensor(min_vr, 
                                              dtype=input.dtype, 
                                              device=torch.device('cuda')))
        

        return torch.as_tensor(min_lambda, dtype=input.dtype, device=torch.device('cuda'))

    @staticmethod
    def backward(ctx, grad_output):
        input, mim_vl, min_vr = ctx.saved_tensors
        grad_weight = None
        grad_input = grad_output * torch.ger(mim_vl, min_vr)

        return grad_input, grad_weight


class JointLoss(nn.Module):
    def __init__(self, opt):
        super(JointLoss, self).__init__()
        self.opt = opt
        self.num_scales = 5

        self.total_loss = None

    def compute_cos_sim_loss(self, gt_n, pred_n, mask):
        cos_criterion= nn.CosineSimilarity(dim=1)
        num_valid_pixels = torch.sum(mask) + EPSILON

        cos_term = cos_criterion(gt_n, pred_n)

        n_term = torch.sum(mask * (1.0 - cos_term))/num_valid_pixels
        return n_term

    def compute_angle_error(self, pred_cam_geo_unit, 
                            pred_up_geo_unit, pred_weights, 
                            targets, stack_error=False):

        gt_up_vector = targets['gt_up_vector'].cuda()

        cos_criterion = nn.CosineSimilarity(dim=0)

        num_pixels = pred_cam_geo_unit.size(2) * pred_cam_geo_unit.size(3)
        num_samples = pred_cam_geo_unit.size(0)

        identity_mat = torch.eye(3).float().cuda()
        identity_mat_rep = identity_mat.unsqueeze(0).repeat(num_samples,1,1)#, requires_grad=False)
        # zeros_mat = Variable(torch.zeros(3).float().cuda(), requires_grad=False)

        weights_n = pred_weights[:, 0:1, :, :].repeat(1,3,1,1)
        weights_u = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)
        weights_t = pred_weights[:, 2:3, :, :].repeat(1,3,1,1)

        pred_cam_n = pred_cam_geo_unit[:, 0:3, :, :] 
        pred_cam_u = pred_cam_geo_unit[:, 3:6, :, :]
        pred_cam_t = pred_cam_geo_unit[:, 6:9, :, :]

        pred_cam_n_w = pred_cam_n * weights_n
        pred_cam_u_w = pred_cam_u * weights_u
        pred_cam_t_w = pred_cam_t * weights_t

        pred_cam_n_w_flat = pred_cam_n_w.view(num_samples, 
                                              pred_cam_n_w.size(1), 
                                              num_pixels)
        pred_cam_u_w_flat = pred_cam_u_w.view(num_samples, 
                                              pred_cam_u_w.size(1), 
                                              num_pixels)
        pred_cam_t_w_flat = pred_cam_t_w.view(num_samples, 
                                              pred_cam_t_w.size(1), 
                                              num_pixels)

        # M * 3 x 3N matrix
        A_w = torch.cat((pred_cam_n_w_flat, pred_cam_u_w_flat, pred_cam_t_w_flat), dim=2)
        
        pred_up_geo_unit_w = pred_weights * pred_up_geo_unit
        pred_up_geo_unit_w_flat = pred_up_geo_unit_w.view(num_samples, pred_up_geo_unit.size(1), num_pixels)
        # M * 1 * 3N
        b_w = torch.cat((pred_up_geo_unit_w_flat[:, 0:1, :], pred_up_geo_unit_w_flat[:, 1:2, :], pred_up_geo_unit_w_flat[:, 2:3, :]), dim=2)

        # M*3*3
        H = torch.bmm(A_w, torch.transpose(A_w, 1, 2))
        # M*3*1
        g = torch.bmm(A_w, torch.transpose(b_w, 1, 2))
        ggT = torch.bmm(g, torch.transpose(g, 1, 2))

        # C_mat = torch.cat( (torch.cat((-A1, -A0), dim=2), torch.cat((identity_mat, zeros_mat), dim=2)), dim=1)
        C_mat = torch.cat( (torch.cat((H, -identity_mat_rep), dim=2), torch.cat((-ggT, H), dim=2)), dim=1)

        if stack_error:
            total_rot_error =  []
            total_roll_error = []
            total_pitch_error = []
        else:
            total_rot_error = 0.0
            total_roll_error = 0.0
            total_pitch_error = 0.0

        for i in range(num_samples):
            est_lambda = torch.eig(C_mat[i, :, :])
            est_lambda = est_lambda[0]

            img_part = est_lambda[:, 1]
            real_part = est_lambda[:, 0]

            min_lambda = torch.min(real_part[torch.abs(img_part.data) < 1e-6])

            est_up_n = torch.matmul(torch.pinverse(H[i, :, :] - min_lambda * identity_mat), g[i, :, :])
            est_up_n_norm = torch.sqrt( torch.sum(est_up_n**2) )
            est_up_n = est_up_n[:, 0]/est_up_n_norm

            up_diff_cos = cos_criterion(est_up_n, gt_up_vector[i, :])

            [pred_roll, pred_pitch] = decompose_up_n(est_up_n.cpu().numpy()) 
            [gt_roll, gt_pitch] = decompose_up_n(gt_up_vector[i, :].cpu().numpy()) 


            if stack_error:
                total_rot_error += [np.arccos(np.clip(up_diff_cos.item(), -1.0, 1.0)) * 180.0/math.pi]
                total_roll_error += [abs(pred_roll - gt_roll)*180.0/math.pi]
                total_pitch_error += [abs(pred_pitch - gt_pitch)*180.0/math.pi]
            else:
                total_rot_error += np.arccos(np.clip(up_diff_cos.item(), -1.0, 1.0)) * 180.0/math.pi
                total_roll_error += abs(pred_roll - gt_roll)*180.0/math.pi
                total_pitch_error += abs(pred_pitch - gt_pitch)*180.0/math.pi

        return total_rot_error, total_roll_error, total_pitch_error

    def compute_normal_gradient_loss(self, gt_n_unit, pred_n_unit, mask):
        n_diff = pred_n_unit - gt_n_unit
        mask_rep = mask.unsqueeze(1).repeat(1, gt_n_unit.size(1), 1, 1)
        # vertical gradient
        v_gradient = torch.abs(n_diff[:, :, :-2,:] - n_diff[:, :, 2:,:])
        v_mask = torch.mul(mask_rep[:, :, :-2,:], mask_rep[:, :, 2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)
        # horizontal gradient
        h_gradient = torch.abs(n_diff[:, :, :, :-2] - n_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_rep[:, :, :, :-2], mask_rep[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON
        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss/(N)
        return gradient_loss

    def normalize_normal(self, normals):
        normals_norm = torch.sqrt( torch.sum(torch.pow(normals , 2) , 1) ).unsqueeze(1).repeat(1,3,1,1) + EPSILON
        return torch.div(normals , normals_norm)

    def normalize_coords(self, coords):
        coord_n = coords[:, 0:3, :, :]
        coord_u = coords[:, 3:6, :, :]
        coord_t = coords[:, 6:9, :, :]

        return torch.cat((self.normalize_normal(coord_n), self.normalize_normal(coord_u), self.normalize_normal(coord_t)), 1)

    def compute_pose_loss(self, gt_up_vector, pred_cam_geo_unit, pred_up_geo_unit, pred_weights, targets=None):
        '''
            solving poses using consraint least sqaure by solving Langrange Multipler directly
        '''
        cos_criterion = nn.CosineSimilarity(dim=0)

        num_pixels = pred_cam_geo_unit.size(2) * pred_cam_geo_unit.size(3)
        num_samples = pred_cam_geo_unit.size(0)

        identity_mat = torch.eye(3).float().cuda()
        identity_mat_rep = identity_mat.unsqueeze(0).repeat(num_samples,1,1)#, requires_grad=False)
        # zeros_mat = Variable(torch.zeros(3).float().cuda(), requires_grad=False)

        weights_n = pred_weights[:, 0:1, :, :].repeat(1,3,1,1)
        weights_u = pred_weights[:, 1:2, :, :].repeat(1,3,1,1)
        weights_t = pred_weights[:, 2:3, :, :].repeat(1,3,1,1)

        pred_cam_n = pred_cam_geo_unit[:, 0:3, :, :] 
        pred_cam_u = pred_cam_geo_unit[:, 3:6, :, :]
        pred_cam_t = pred_cam_geo_unit[:, 6:9, :, :]

        pred_cam_n_w = pred_cam_n * weights_n
        pred_cam_u_w = pred_cam_u * weights_u
        pred_cam_t_w = pred_cam_t * weights_t

        pred_cam_n_w_flat = pred_cam_n_w.view(num_samples, 
                                              pred_cam_n_w.size(1), 
                                              num_pixels)
        pred_cam_u_w_flat = pred_cam_u_w.view(num_samples, 
                                              pred_cam_u_w.size(1), 
                                              num_pixels)
        pred_cam_t_w_flat = pred_cam_t_w.view(num_samples, 
                                              pred_cam_t_w.size(1), 
                                              num_pixels)

        # M * 3 x 3N matrix
        A_w = torch.cat((pred_cam_n_w_flat, 
                         pred_cam_u_w_flat, 
                         pred_cam_t_w_flat), dim=2)
        
        pred_up_geo_unit_w = pred_weights * pred_up_geo_unit
        pred_up_geo_unit_w_flat = pred_up_geo_unit_w.view(num_samples, 
                                                          pred_up_geo_unit.size(1), 
                                                          num_pixels)
        # M * 1 * 3N
        b_w = torch.cat((pred_up_geo_unit_w_flat[:, 0:1, :], 
                         pred_up_geo_unit_w_flat[:, 1:2, :], 
                         pred_up_geo_unit_w_flat[:, 2:3, :]), dim=2)

        # M*3*3
        H = torch.bmm(A_w, torch.transpose(A_w, 1, 2))
        # M*3*1
        g = torch.bmm(A_w, torch.transpose(b_w, 1, 2))
        ggT = torch.bmm(g, torch.transpose(g, 1, 2))

        # A0 = torch.bmm(H, H) - ggT
        # A1 = -2.0 * H

        # C_mat = torch.cat( (torch.cat((-A1, -A0), dim=2), torch.cat((identity_mat, zeros_mat), dim=2)), dim=1)
        C_mat = torch.cat( (torch.cat((H, -identity_mat_rep), dim=2), 
                            torch.cat((-ggT, H), dim=2)), dim=1)


        pose_term = 0.0

        for i in range(num_samples):

            if self.opt.backprop_eig > EPSILON:
                min_lambda = ExtractSMinEigenValue.apply(C_mat[i, :, :])
            else:
                est_lambda = torch.eig(C_mat[i, :, :])
                est_lambda = est_lambda[0]

                img_part = est_lambda[:, 1]
                real_part = est_lambda[:, 0]

                min_lambda = torch.min(real_part[torch.abs(img_part.data) < 1e-6]).item()

            est_up_n = torch.matmul(torch.pinverse(H[i, :, :] - min_lambda * identity_mat), 
                                    g[i, :, :])

            up_diff_cos = cos_criterion(est_up_n[:, 0], 
                                        gt_up_vector[i, :])

            # print('min_lambda ', min_lambda)
            # print('i %d up_diff_cos %f ang_diff %f'%(i, up_diff_cos.item(), torch.acos(torch.clamp(up_diff_cos, -1.0, 1.0)).item() * 180/math.pi))

            if up_diff_cos.item() < (1.0 - 1e-6):
                up_diff_angle_rad = torch.acos(up_diff_cos)
            else:
                up_diff_angle_rad = 1.0 - up_diff_cos
            
            pose_term += self.opt.w_pose * up_diff_angle_rad/float(num_samples)

        # print('pose_term %f'%pose_term.item())

        return pose_term
        

    def rotate_normal(self, pred_global_n_unit, pred_rot):
        num_samples = pred_global_n_unit.size(0)
        num_c = pred_global_n_unit.size(1)
        num_pixels = pred_global_n_unit.size(2) * pred_global_n_unit.size(3)

        pred_global_n_unit_flat = pred_global_n_unit.view(num_samples, 
                                                          num_c, 
                                                          num_pixels)

        pred_cam_n_unit_flat = torch.bmm(pred_rot, pred_global_n_unit_flat)
        pred_cam_n_unit = pred_cam_n_unit_flat.view(num_samples, num_c, 
                                                    pred_global_n_unit.size(2), 
                                                    pred_global_n_unit.size(3))

        return pred_cam_n_unit


    def __call__(self, input_images, pred_cam_geo_unit, pred_up_geo_unit, pred_weights, targets):
        gt_up_vector = Variable(targets['gt_up_vector'].cuda(), 
                                requires_grad=False) 
        gt_rp = Variable(targets['gt_rp'].cuda(), 
                         requires_grad=False)

        gt_upright_geo = Variable(targets['upright_geo'].cuda(), 
                                  requires_grad=False)
        gt_cam_geo = Variable(targets['cam_geo'].cuda(), 
                              requires_grad=False)
        
        gt_mask = Variable(targets['gt_mask'].cuda(), 
                           requires_grad=False)
        
        total_loss = 0.

        if self.opt.w_pose > EPSILON:
            pose_term = self.compute_pose_loss(gt_up_vector, 
                                               pred_cam_geo_unit, 
                                               pred_up_geo_unit, 
                                               pred_weights)
            total_loss += pose_term
        else:
            pose_term = torch.tensor(0.).cuda()

        if self.opt.w_cam > EPSILON:
            cam_geo_term = 0.0
            for i in range(0, 3):
                cam_geo_term += self.opt.w_cam * self.compute_cos_sim_loss(gt_cam_geo[:, i*3:(i+1)*3, :, :], 
                                                                           pred_cam_geo_unit[:, i*3:(i+1)*3, :, :], 
                                                                           gt_mask)

            cam_geo_term = cam_geo_term/3.0
            print('cam_geo_term ', cam_geo_term.item())
            total_loss = total_loss + cam_geo_term

            if self.opt.w_grad > EPSILON:
                cam_grad_term = 0.
                for j in range(self.num_scales):
                    stride = 2**j
                    cam_grad_term += self.opt.w_grad * self.opt.w_cam * self.compute_normal_gradient_loss(gt_cam_geo[:, :, ::stride, ::stride], 
                                                                                                          pred_cam_geo_unit[:, :, ::stride, ::stride], 
                                                                                                          gt_mask[:, ::stride, ::stride])

                print('cam_grad_term ', cam_grad_term.item())
                total_loss += cam_grad_term

        else:
            cam_geo_term = torch.tensor(0.).cuda()

        if self.opt.w_up > EPSILON:
            upright_geo_term = self.opt.w_up * self.compute_cos_sim_loss(gt_upright_geo, 
                                                                         pred_up_geo_unit, 
                                                                         gt_mask)

            print('upright_geo_term ', upright_geo_term.item())
            total_loss = total_loss + upright_geo_term

            if self.opt.w_grad > EPSILON:
                upright_grad_term = 0.
                for j in range(self.num_scales):
                    stride = 2**j
                    upright_grad_term += self.opt.w_grad * self.opt.w_up * self.compute_normal_gradient_loss(gt_upright_geo[:, :, ::stride, ::stride], 
                                                                                                             pred_up_geo_unit[:, :, ::stride, ::stride], 
                                                                                                             gt_mask[:, ::stride, ::stride])

                print('upright_grad_term ', upright_grad_term.item())
                total_loss += upright_grad_term
        else:
            upright_n_term = torch.tensor(0.).cuda()


        self.total_loss = total_loss

        return total_loss.item(), cam_geo_term.item(), upright_geo_term.item(), pose_term.item()

    def get_loss_var(self):
        return self.total_loss

