import torch.utils.data as data
import pickle
import PIL
import numpy as np
import torch
import os
import math, random
import os.path
import sys
import cv2
import skimage
from skimage.transform import rotate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WALL_THREAHOD = 5e-2
GC_THRESHOLD = 1e-1
VIZ = False

def decompose_rotation(R):

    pitch_2 = math.atan2(-R[2,0], 
                         math.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll_2 = math.atan2(R[2, 1]/math.cos(pitch_2), 
                        R[2, 2]/math.cos(pitch_2))
    yaw_2 = math.atan2(R[1,0]/math.cos(pitch_2), 
                       R[0,0]/math.cos(pitch_2))

    return [roll_2, pitch_2,yaw_2]

def decompose_up_n(up_n):
    pitch = - math.asin(up_n[0])

    sin_roll = up_n[1]/math.cos(pitch)

    roll = math.asin(sin_roll)
    return roll, pitch

def get_xy_vector_from_rp(roll, pitch):
    rx = np.array( (math.cos(pitch) * math.cos(roll) , 0.0, math.sin(pitch) ))
    ry = np.array( (0.0, math.cos(roll), -math.sin(roll) ))

    return rx, ry

def make_dataset(list_name):
    text_file = open(list_name, "r")
    images_list = text_file.readlines()
    text_file.close()
    return images_list

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


class InteriorNetRyFolder(data.Dataset):

    def __init__(self, opt, list_path, is_train):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.list_path = list_path
        self.img_list = img_list
        self.opt = opt
        self.input_width = 384
        self.input_height = 288

        self.is_train = is_train

        self.rot_range = 10
        self.reshape = False
        self.lr_threshold = 4.
        self.fx = 600.
        self.fy = 600.

    def load_imgs(self, img_path, normal_path, rot_path):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        normal = (np.float32(cv2.imread(normal_path, -1))/65535. * 2.0) - 1.0
        normal = normal[:, :, ::-1]

        h, w, c = normal.shape

        cam_normal = normal[:, :w//2, :]
        global_normal = normal[:, w//2:, :]

        mask = np.float32(np.linalg.norm(cam_normal, axis=-1) > 0.9) * np.float32(np.linalg.norm(cam_normal, axis=-1) < 1.1)
        R_g_c = np.identity(3)

        with open(rot_path, 'r') as f:
            rot_row = f.readlines()

            for i in range(3):
                r1, r2, r3 = rot_row[i].split()
                R_g_c[i, :] = np.array((np.float32(r1), np.float32(r2), np.float32(r3)))

        return {'img': img, 
                'cam_normal':cam_normal, 
                'global_normal': global_normal,
                'mask':mask,
                'R_g_c': R_g_c}

 
    def resize_imgs(self, train_data, resized_width, resized_height):
        train_data['img'] = cv2.resize(train_data['img'], 
                                       (resized_width, resized_height), 
                                       interpolation=cv2.INTER_AREA)
        train_data['cam_normal'] = cv2.resize(train_data['cam_normal'], 
                                       (resized_width, resized_height), 
                                       interpolation=cv2.INTER_NEAREST)
        train_data['global_normal'] = cv2.resize(train_data['global_normal'], 
                                       (resized_width, resized_height), 
                                       interpolation=cv2.INTER_NEAREST)
        train_data['mask'] = cv2.resize(train_data['mask'], 
                                       (resized_width, resized_height), 
                                       interpolation=cv2.INTER_NEAREST)

        return train_data

    def crop_imgs(self, train_data, start_x, start_y, crop_w, crop_h):
        train_data['img'] = train_data['img'][start_y:start_y+crop_h, 
                                              start_x:start_x+crop_w, :]
        train_data['cam_normal'] = train_data['cam_normal'][start_y:start_y+crop_h, 
                                                            start_x:start_x+crop_w, :]
        train_data['global_normal'] = train_data['global_normal'][start_y:start_y+crop_h, 
                                                                  start_x:start_x+crop_w, :]
        train_data['mask'] = train_data['mask'][start_y:start_y+crop_h, 
                                                start_x:start_x+crop_w]

        return train_data

    def load_precomputed_crop_hw(self, normal_path):
        crop_hw_path = normal_path.replace('normal_pair', 'precomputed_crop_hw')[:-4] + '.txt'

        with open(crop_hw_path, 'r') as f:
            crop_hw = f.readlines()
            crop_h, crop_w = crop_hw[0].split()


        return int(crop_h), int(crop_w)

    def rotate_normal(self, R, normal):
        normal_rot = np.dot(R, np.reshape(normal, (-1, 3)).T)
        normal_rot = np.reshape(normal_rot.T, (normal.shape[0], normal.shape[1], 3))
        normal_rot = normal_rot/(np.maximum(np.linalg.norm(normal_rot, axis=2, keepdims=True), 1e-8))
        normal_rot = np.clip(normal_rot, -1.0, 1.0)
        return normal_rot

    def create_geo_ry(self, cam_normal, global_normal, R_gc):

        wall_mask = np.abs(global_normal[:, :, 2]) < WALL_THREAHOD #* mask

        upright_u_y = global_normal[:, :, 0].copy()
        upright_u_z = global_normal[:, :, 2].copy()
        
        upright_u_z[wall_mask] = 0.0
        upright_u_y[wall_mask] = 1.0

        global_u_unit = np.stack((-upright_u_z, np.zeros_like(upright_u_z), upright_u_y), axis=2)
        global_u_unit = global_u_unit/(np.maximum(np.linalg.norm(global_u_unit, axis=2, keepdims=True), 1e-8))

        cam_u_unit = self.rotate_normal(R_gc.T, global_u_unit) 

        global_t_unit = np.cross(global_u_unit, global_normal)
        global_t_unit = global_t_unit/(np.maximum(np.linalg.norm(global_t_unit, axis=2, keepdims=True), 1e-8))

        cam_t_unit = self.rotate_normal(R_gc.T, global_t_unit) 

        cam_geo = np.concatenate((cam_normal, cam_u_unit, cam_t_unit), axis=2)
        global_geo = np.concatenate((global_normal[:, :, 2:3], global_u_unit[:, :, 2:3], global_t_unit[:, :, 2:3]), axis=2)
        global_geo = global_geo/(np.maximum(np.linalg.norm(global_geo, axis=2, keepdims=True), 1e-8))

        return cam_geo, global_geo#, np.float32(wall_mask)


    def create_geo_rz(self, cam_normal, global_normal, R_gc):

        gc_mask = np.abs(np.abs(global_normal[:, :, 2]) - 1.0) < GC_THRESHOLD #* mask

        global_t_x = global_normal[:, :, 0].copy()
        global_t_y = global_normal[:, :, 1].copy()
    
        global_t_x[gc_mask] = -1.0
        global_t_y[gc_mask] = 0.0

        global_t_unit = np.stack( (-global_t_y, global_t_x, np.zeros_like(global_t_x)), axis=2)
        global_t_unit = global_t_unit/(np.maximum(np.linalg.norm(global_t_unit, axis=2, keepdims=True), 1e-8))

        cam_t_unit = self.rotate_normal(R_gc.T, global_t_unit) 

        global_u_unit = np.cross(global_normal, global_t_unit)
        global_u_unit = global_u_unit/(np.maximum(np.linalg.norm(global_u_unit, axis=2, keepdims=True), 1e-8))

        cam_u_unit = self.rotate_normal(R_gc.T, global_u_unit) 
        cam_geo = np.concatenate((cam_normal, cam_u_unit, cam_t_unit), axis=2)

        global_geo = np.concatenate((global_normal[:, :, 2:3], 
                                     global_u_unit[:, :, 2:3], 
                                     global_t_unit[:, :, 2:3]), axis=2)
        
        global_geo = global_geo/(np.maximum(np.linalg.norm(global_geo, axis=2, keepdims=True), 1e-8))

        return cam_geo, global_geo#, np.float32(wall_mask)


    def __getitem__(self, index):
        targets_1 = {}

        normal_path = self.img_list[index].rstrip()#.split()

        img_path = normal_path.replace('normal_pair', 'rgb') #+ '.png'
        rot_path = normal_path.replace('normal_pair', 'gt_poses')[:-4] + '.txt'

        train_data = self.load_imgs(img_path, normal_path, rot_path)

        original_h, original_w = train_data['img'].shape[0], train_data['img'].shape[1]

        if self.is_train:
            crop_h = random.randint(380, original_h)
            crop_w = int(round(crop_h*float(original_w)/float(original_h)))

            start_y = random.randint(0, original_h - crop_h)
            start_x = random.randint(0, original_w - crop_w)

            train_data = self.crop_imgs(train_data, 
                                        start_x, start_y, 
                                        crop_w, crop_h)
            train_data = self.resize_imgs(train_data, 
                                        self.input_width, 
                                        self.input_height)

        else:
            crop_h, crop_w = self.load_precomputed_crop_hw(normal_path)
            start_y = int((original_h - crop_h)/2)
            start_x = int((original_w - crop_w)/2)

            train_data = self.crop_imgs(train_data, start_x, start_y, crop_w, crop_h)
            train_data = self.resize_imgs(train_data, self.input_width, self.input_height)

        ratio_x = float(train_data['img'].shape[1])/float(crop_w)
        ratio_y = float(train_data['img'].shape[0])/float(crop_h)

        fx = self.fx * ratio_x
        fy = self.fy * ratio_y

        img_1 = np.float32(train_data['img'])/255.0
        cam_normal = train_data['cam_normal']
        R_g_c = train_data['R_g_c']

        global_normal = train_data['global_normal']
        upright_normal = self.rotate_normal(R_g_c, cam_normal)

        mask = train_data['mask']
        gt_up_vector = R_g_c[2, :]
        [gt_roll, gt_pitch, gt_yaw]= decompose_rotation(R_g_c)
        gt_rp = np.array([gt_roll, gt_pitch]) 

        cam_geo, upright_geo = self.create_geo_ry(cam_normal, upright_normal, R_g_c)
        
        if VIZ:
            save_img_name = 'imgs/' + normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1]
            save_img_name = save_img_name[:-4] + '.jpg'
            skimage.io.imsave(save_img_name, img_1)
                
            save_n_name = save_img_name[:-4] + '_n.jpg'
            cam_n_rgb = (cam_geo[:, :, 0:3] + 1.0)/2.
            skimage.io.imsave(save_n_name, cam_n_rgb)

            save_u_name = save_img_name[:-4] + '_u.jpg'
            cam_u_rgb = (cam_geo[:, :, 3:6] + 1.0)/2.
            skimage.io.imsave(save_u_name, cam_u_rgb)

            save_t_name = save_img_name[:-4] + '_t.jpg'
            cam_t_rgb = (cam_geo[:, :, 6:9] + 1.0)/2.
            skimage.io.imsave(save_t_name, cam_t_rgb)

            upright_v_name = save_img_name[:-4] + '_v.jpg'
            upright_geo_rgb = (upright_geo + 1.0)/2.
            skimage.io.imsave(upright_v_name, upright_geo_rgb)

            print('%s from rotation matrix: roll %f, pitch %f, yaw %f'%(img_path, math.degrees(gt_roll), math.degrees(gt_pitch), math.degrees(gt_yaw)))
            plt.figure(figsize=(12, 6))
            plt.subplot(2,4,1)
            plt.imshow(img_1) 

            plt.subplot(2,4,5)
            plt.imshow(mask, cmap='gray')

            plt.subplot(2,4,2)
            plt.imshow((cam_geo[:, :, 0:3] + 1.0)/2.) 

            plt.subplot(2,4,6)
            plt.imshow((upright_geo[:, :, 0]+1.)/2.0, cmap='gray') 

            plt.subplot(2,4,3)
            plt.imshow((cam_geo[:, :, 3:6] + 1.0)/2.) 

            plt.subplot(2,4,7)
            plt.imshow( (upright_geo[:, :, 1]+1.0)/2., cmap='gray') 

            plt.subplot(2,4,4)
            plt.imshow((cam_geo[:, :, 6:9] + 1.0)/2.) 

            plt.subplot(2,4,8)
            plt.imshow( (upright_geo+1.0)/2.) 


            plt.savefig(normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1])
            # combine_gt = np.concatenate((img_1, (cam_nu[:, :, 0:3] + 1.0)/2., (upright_nu[:, :, 0:3]+1.)/2.0, (cam_nu[:, :, 3:6] + 1.0)/2., (cam_nu[:, :, 3:6] + 1.0)/2.), axis=1) 

            # save_img = np.uint16( np.clip(np.round(65535.0 * combine_gt), 0., 65535.))
            # cv2.imwrite(normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1], save_img[:, :, ::-1])

            print('InteriorNet train we are good')
            sys.exit()  


        final_img = torch.from_numpy(np.ascontiguousarray(img_1).transpose(2,0,1)).contiguous().float()
        targets_1['cam_geo'] = torch.from_numpy(np.ascontiguousarray(cam_geo).transpose(2,0,1)).contiguous().float()
        targets_1['upright_geo'] = torch.from_numpy(np.ascontiguousarray(upright_geo).transpose(2,0,1)).contiguous().float()
        
        targets_1['gt_mask'] = torch.from_numpy(np.ascontiguousarray(mask)).contiguous().float()

        targets_1['gt_rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
        targets_1['R_g_c'] = torch.from_numpy(np.ascontiguousarray(R_g_c)).contiguous().float()
        targets_1['gt_up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()

        targets_1['img_path'] = img_path
        targets_1['normal_path'] = normal_path
        targets_1['fx'] = fx
        targets_1['fy'] = fy

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


# class SUN360Folder(data.Dataset):

#     def __init__(self, opt, list_path, is_train):
#         img_list = make_dataset(list_path)
#         if len(img_list) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#         # self.img_dir = img_dir
#         self.list_path = list_path
#         self.img_list = img_list
#         self.opt = opt
#         self.input_width = 384
#         self.input_height = 288

#         self.is_train = is_train
#         self.brightness_factor = 0.1
#         self.contrast_factor = 0.1
#         self.saturation_factor = 0.1
#         self.rot_range = 20
#         self.reshape = False
#         self.lr_threshold = 4.

#     def load_imgs(self, img_path, rot_path):
#         img = cv2.imread(img_path)


#         try:
#             img = img[:,:,::-1]
#         except:
#             print(img_path)
#             sys.exit()


#         R_g_c = np.identity(3)

#         R_g_c = np.identity(3)

#         with open(rot_path, 'r') as f:
#             rot_row = f.readlines()

#             for i in range(3):
#                 r1, r2, r3 = rot_row[i].split()
#                 R_g_c[i, :] = np.array((np.float32(r1), np.float32(r2), np.float32(r3)))

#         return {'img': img,
#                 'R_g_c': R_g_c}

#     def rotate_imgs(self, train_data, random_angle):
#         # first rotate input image
#         # then compute rotation matrix to transform camera normal
#         R_r_c = np.identity(3)
#         random_radius = - random_angle/180.0 * math.pi
#         R_r_c[0,0] = math.cos(random_radius)
#         R_r_c[0,2] = math.sin(random_radius)
#         R_r_c[2,0] = -math.sin(random_radius)
#         R_r_c[2,2] = math.cos(random_radius)

#         cam_normal_rot = np.dot(R_r_c, np.reshape(train_data['cam_normal'], (-1, 3)).T)
#         cam_normal_rot = np.reshape(cam_normal_rot.T, (train_data['cam_normal'].shape[0], train_data['cam_normal'].shape[1], 3))

#         train_data['R_g_c'] = np.dot(train_data['R_g_c'], R_r_c.T)

#         resize = False

#         train_data['img'] = rotate(train_data['img'], random_angle, order=1, resize=resize)
#         train_data['cam_normal'] = rotate(cam_normal_rot, random_angle, order=0, resize=resize)
#         train_data['upright_normal'] = rotate(train_data['upright_normal'], random_angle, order=0, resize=resize)
#         train_data['mask'] = rotate(train_data['mask'], random_angle, order=0, resize=resize)

#         return train_data

#     def resize_imgs(self, train_data, resized_width, resized_height):
#         train_data['img'] = cv2.resize(train_data['img'], (resized_width, resized_height), interpolation=cv2.INTER_AREA)

#         return train_data

#     def crop_imgs(self, train_data, start_x, start_y, crop_w, crop_h):
#         train_data['img'] = train_data['img'][start_y:start_y+crop_h, start_x:start_x+crop_w, :]

#         return train_data

#     def load_intrinsic(self, intrinsic_path):
#         intrinsic = np.identity(3)

#         with open(intrinsic_path, 'r') as f:
#             rot_row = f.readlines()

#             for i in range(3):
#                 r1, r2, r3 = rot_row[i].split()
#                 intrinsic[i, :] = np.array((np.float32(r1), np.float32(r2), np.float32(r3)))

#         return intrinsic[0, 0]/2.0, intrinsic[1, 1]/2.0

#     def __getitem__(self, index):
#         targets_1 = {}

#         img_path = self.img_list[index].rstrip()#.split()

#         poses_path = img_path.replace('rgb/', 'poses/').replace('.png', '_true_camera_rotation.txt')
#         intrinsic_path = img_path.replace('rgb/', 'intrinsic/').replace('.png', '_true_camera_intrinsic.txt')

#         train_data = self.load_imgs(img_path, poses_path)       
#         original_h, original_w = train_data['img'].shape[0], train_data['img'].shape[1]
#         fx_o, fy_o = self.load_intrinsic(intrinsic_path)

#         train_data = self.resize_imgs(train_data, self.input_width, self.input_height)

#         ratio_x = float(train_data['img'].shape[1])/float(original_w)
#         ratio_y = float(train_data['img'].shape[0])/float(original_h)

#         fx = fx_o * ratio_x
#         fy = fy_o * ratio_y

#         img_h, img_w = train_data['img'].shape[0], train_data['img'].shape[1]

#         img_1 = np.float32(train_data['img'])/255.0
#         mask = np.float32(np.mean(img_1, -1) > 1e-4)
#         R_g_c = train_data['R_g_c']

#         [gt_roll, gt_pitch, gt_yaw]= decompose_rotation(R_g_c)

#         gt_vfov = 2 * math.atan(float(img_h)/(2*fy))
#         gt_up_vector = R_g_c[2, :]

#         gt_rp = np.array([gt_roll, gt_pitch]) 

#         if VIZ:
#             hl_left, hl_right = getHorizonLineFromAngles(gt_pitch, gt_roll, gt_vfov, img_h, img_w)

#             slope = np.arctan(hl_right - hl_left)
#             midpoint = (hl_left + hl_right) / 2.0
#             offset = (midpoint - 0.5) / np.sqrt( 1 + (hl_right - hl_left)**2 )

#             slope_idx = np.clip(np.digitize(slope, slope_bins), 0, len(slope_bins)-1)
#             offset_idx = np.clip(np.digitize(offset, offset_bins), 0, len(offset_bins)-1)

#             print('%s roll %f, pitch %f, yaw %f vfov %f'%(img_path, math.degrees(gt_roll), math.degrees(gt_pitch), math.degrees(gt_yaw), math.degrees(gt_vfov)))

#             plt.figure(figsize=(10, 6))
#             plt.subplot(2,1,1)
#             plt.imshow(img_1) 
#             plt.subplot(2,1,2)
#             plt.imshow(mask, cmap='gray')

#             # plt.subplot(2,2,3)
#             # plt.imshow((cam_normal+1.)/2.0) 

#             # plt.subplot(2,2,4)
#             # plt.imshow((upright_normal+1.)/2.0) 

#             plt.savefig(img_path.split('/')[-1])
#             print('train we are good MP')
#             sys.exit()


#         final_img = torch.from_numpy(np.ascontiguousarray(img_1).transpose(2,0,1)).contiguous().float()
#         targets_1['gt_mask'] = torch.from_numpy(np.ascontiguousarray(mask)).contiguous().float()
#         targets_1['R_g_c'] = torch.from_numpy(np.ascontiguousarray(R_g_c)).contiguous().float()
#         targets_1['gt_rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
#         targets_1['gt_up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()
#         targets_1['fx'] = torch.from_numpy(np.ascontiguousarray(fx)).contiguous().float()
#         targets_1['fy'] = torch.from_numpy(np.ascontiguousarray(fy)).contiguous().float()
#         targets_1['img_path'] = img_path
#         return final_img, targets_1

#     def __len__(self):
#         return len(self.img_list)


class ScanNetFolder(data.Dataset):

    def __init__(self, opt, list_path, is_train):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.list_path = list_path
        self.img_list = img_list
        self.opt = opt
        self.input_width = 384
        self.input_height = 288

        self.is_train = is_train
        self.rot_range = 10
        self.reshape = False
        self.lr_threshold = 4.

    def load_imgs(self, img_path, normal_path, rot_path, intrinsic_path):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        normal = (np.float32(cv2.imread(normal_path, -1))/65535. * 2.0) - 1.0
        cam_normal = normal[:, :, ::-1]

        mask = np.float32(np.linalg.norm(cam_normal, axis=-1) > 0.9) * np.float32(np.linalg.norm(cam_normal, axis=-1) < 1.1) #* np.float32(np.max(img,-1) > 1e-3)

        R_g_c = np.identity(3)

        with open(rot_path, 'r') as f:
            rot_row = f.readlines()

            for i in range(3):
                r1, r2, r3 = rot_row[i].split()
                R_g_c[i, :] = np.array((np.float32(r1), np.float32(r2), np.float32(r3)))

        intrinsic = np.identity(3)

        with open(intrinsic_path, 'r') as f:
            intrinsic_row = f.readlines()

            for i in range(2):
                i1, i2, i3 = intrinsic_row[i].split()
                intrinsic[i, :] = np.array((np.float32(i1), np.float32(i2), np.float32(i3)))

        upright_normal = self.rotate_normal(R_g_c, cam_normal)

        return {'img': img, 
                'cam_normal':cam_normal, 
                'upright_normal':upright_normal,
                'mask':mask,
                'R_g_c': R_g_c, 
                'intrinsic':intrinsic}

    def resize_imgs(self, train_data, resized_width, resized_height):
        train_data['img'] = cv2.resize(train_data['img'], 
                                      (resized_width, resized_height), 
                                      interpolation=cv2.INTER_AREA)
        train_data['cam_normal'] = cv2.resize(train_data['cam_normal'], 
                                      (resized_width, resized_height), 
                                      interpolation=cv2.INTER_NEAREST)
        train_data['upright_normal'] = cv2.resize(train_data['upright_normal'], 
                                      (resized_width, resized_height), 
                                      interpolation=cv2.INTER_NEAREST)
        train_data['mask'] = cv2.resize(train_data['mask'], 
                                      (resized_width, resized_height), 
                                      interpolation=cv2.INTER_NEAREST)

        return train_data

    def crop_imgs(self, train_data, start_x, start_y, crop_w, crop_h):
        train_data['img'] = train_data['img'][start_y:start_y+crop_h, start_x:start_x+crop_w, :]
        train_data['cam_normal'] = train_data['cam_normal'][start_y:start_y+crop_h, start_x:start_x+crop_w, :]
        train_data['upright_normal'] = train_data['upright_normal'][start_y:start_y+crop_h, start_x:start_x+crop_w, :]
        train_data['mask'] = train_data['mask'][start_y:start_y+crop_h, start_x:start_x+crop_w]

        return train_data

    def load_precomputed_crop_hw(self, normal_path):
        crop_hw_path = normal_path.replace('normal_pair', 'precomputed_crop_hw')[:-4] + '.txt'

        with open(crop_hw_path, 'r') as f:
            crop_hw = f.readlines()
            crop_h, crop_w = crop_hw[0].split()

        return int(crop_h), int(crop_w)

    def rotate_imgs(self, train_data, random_angle):
        # first rotate input image
        # then compute rotation matrix to transform camera normal
        R_r_c = np.identity(3)
        random_radius = - random_angle/180.0 * math.pi
        R_r_c[0,0] = math.cos(random_radius)
        R_r_c[0,2] = math.sin(random_radius)
        R_r_c[2,0] = -math.sin(random_radius)
        R_r_c[2,2] = math.cos(random_radius)

        cam_normal_rot = np.dot(R_r_c, np.reshape(train_data['cam_normal'], (-1, 3)).T)
        cam_normal_rot = np.reshape(cam_normal_rot.T, (train_data['cam_normal'].shape[0], train_data['cam_normal'].shape[1], 3))

        train_data['R_g_c'] = np.dot(train_data['R_g_c'], R_r_c.T)

        resize = False

        train_data['img'] = rotate(train_data['img'], random_angle, order=1, resize=resize)
        train_data['cam_normal'] = rotate(cam_normal_rot, random_angle, order=0, resize=resize)
        train_data['upright_normal'] = rotate(train_data['upright_normal'], random_angle, order=0, resize=resize)
        train_data['mask'] = rotate(train_data['mask'], random_angle, order=0, resize=resize)

        return train_data

    def rotate_normal(self, R, normal):
        normal_rot = np.dot(R, np.reshape(normal, (-1, 3)).T)
        normal_rot = np.reshape(normal_rot.T, (normal.shape[0], normal.shape[1], 3))
        normal_rot = normal_rot/(np.maximum(np.linalg.norm(normal_rot, axis=2, keepdims=True), 1e-8))
        normal_rot = np.clip(normal_rot, -1.0, 1.0)
        return normal_rot

    def create_geo_ry(self, cam_normal, global_normal, R_gc):
        wall_mask = np.abs(global_normal[:, :, 2]) < WALL_THREAHOD

        upright_u_y = global_normal[:, :, 0].copy()
        upright_u_z = global_normal[:, :, 2].copy()
        
        upright_u_z[wall_mask] = 0.0
        upright_u_y[wall_mask] = 1.0

        global_u_unit = np.stack((-upright_u_z, np.zeros_like(upright_u_z), upright_u_y), axis=2)
        global_u_unit = global_u_unit/(np.maximum(np.linalg.norm(global_u_unit, axis=2, keepdims=True), 1e-8))

        cam_u_unit = self.rotate_normal(R_gc.T, global_u_unit) 

        global_t_unit = np.cross(global_u_unit, global_normal)
        global_t_unit = global_t_unit/(np.maximum(np.linalg.norm(global_t_unit, axis=2, keepdims=True), 1e-8))

        cam_t_unit = self.rotate_normal(R_gc.T, global_t_unit) 

        cam_ut = np.concatenate((cam_normal, cam_u_unit, cam_t_unit), axis=2)
        # global_ut = np.concatenate((global_normal, global_u_unit, global_t_unit), axis=2)
        global_ut = np.concatenate((global_normal[:, :, 2:3], global_u_unit[:, :, 2:3], global_t_unit[:, :, 2:3]), axis=2)
        # global_ut = global_ut/(np.maximum(np.linalg.norm(global_ut, axis=2, keepdims=True), 1e-8))

        return cam_ut, global_ut#, np.float32(wall_mask)


    def __getitem__(self, index):
        targets_1 = {}

        normal_path = self.img_list[index].rstrip()#.split()

        img_path = normal_path.replace('normal_pair', 'rgb') #+ '.png'
        rot_path = normal_path.replace('normal_pair', 'gt_poses')[:-4] + '.txt'
        intrinsic_path = '/'.join(normal_path.replace('normal_pair', 'intrinsic').split('/')[:-1]) + '/intrinsic_resized.txt'

        train_data = self.load_imgs(img_path, normal_path, rot_path, intrinsic_path)

        original_h, original_w = train_data['img'].shape[0], train_data['img'].shape[1]

        if self.is_train:
            crop_h = random.randint(380, original_h)
            crop_w = int(round(crop_h*float(original_w)/float(original_h)))

            start_y = random.randint(0, original_h - crop_h)
            start_x = random.randint(0, original_w - crop_w)

            train_data = self.crop_imgs(train_data, start_x, start_y, crop_w, crop_h)
            train_data = self.resize_imgs(train_data, self.input_width, self.input_height)

        else:
            crop_h, crop_w = self.load_precomputed_crop_hw(normal_path)
            start_y = int((original_h - crop_h)/2)#random.randint(0, original_h - crop_h)
            start_x = int((original_w - crop_w)/2)#random.randint(0, original_w - crop_w)

            train_data = self.crop_imgs(train_data, start_x, start_y, crop_w, crop_h)
            train_data = self.resize_imgs(train_data, self.input_width, self.input_height)

        ratio_x = float(train_data['img'].shape[1])/float(crop_w)
        ratio_y = float(train_data['img'].shape[0])/float(crop_h)

        intrinsic = train_data['intrinsic']

        fx = intrinsic[0, 0] * ratio_x
        fy = intrinsic[1, 1] * ratio_y

        img_1 = np.float32(train_data['img'])/255.0
        cam_normal = train_data['cam_normal']
        upright_normal = train_data['upright_normal']
        mask = train_data['mask']
        R_g_c = train_data['R_g_c']
        gt_up_vector = R_g_c[2, :]
        [gt_roll, gt_pitch, gt_yaw] = decompose_rotation(R_g_c)

        gt_rp = np.array([gt_roll, gt_pitch]) 

        cam_geo, upright_geo = self.create_geo_ry(cam_normal, upright_normal, R_g_c)

        if VIZ:
            save_img_name = 'imgs/' + normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1]
            save_img_name = save_img_name[:-4] + '.jpg'
            skimage.io.imsave(save_img_name, img_1)
            # sys.exit()
                
            save_n_name = save_img_name[:-4] + '_n.jpg'
            cam_n_rgb = (cam_geo[:, :, 0:3] + 1.0)/2.
            skimage.io.imsave(save_n_name, cam_n_rgb)

            save_u_name = save_img_name[:-4] + '_u.jpg'
            cam_u_rgb = (cam_geo[:, :, 3:6] + 1.0)/2.
            skimage.io.imsave(save_u_name, cam_u_rgb)

            save_t_name = save_img_name[:-4] + '_t.jpg'
            cam_t_rgb = (cam_geo[:, :, 6:9] + 1.0)/2.
            skimage.io.imsave(save_t_name, cam_t_rgb)

            upright_v_name = save_img_name[:-4] + '_v.jpg'
            upright_geo_rgb = (upright_geo + 1.0)/2.
            skimage.io.imsave(upright_v_name, upright_geo_rgb)

            print('%s from rotation matrix: roll %f, pitch %f, yaw %f'%(img_path, math.degrees(gt_roll), math.degrees(gt_pitch), math.degrees(gt_yaw)))
            print('%s from up vector: roll %f, pitch %f'%(img_path, math.degrees(up_roll), math.degrees(up_pitch)))
            
            gc_mask = np.abs(np.abs(upright_normal[:, :, 2]) - 1.0) < GC_THRESHOLD #* mask
            gc_mask = remove_small_objects(gc_mask, min_size=100)

            plt.figure(figsize=(12, 6))
            plt.subplot(2,4,1)
            plt.imshow(img_1) 

            plt.subplot(2,4,5)
            plt.imshow(mask, cmap='gray')

            plt.subplot(2,4,2)
            plt.imshow((cam_geo[:, :, 0:3] + 1.0)/2.) 

            plt.subplot(2,4,6)
            plt.imshow((upright_geo[:, :, 0]+1.)/2.0, cmap='gray') 

            plt.subplot(2,4,3)
            plt.imshow((cam_geo[:, :, 3:6] + 1.0)/2.) 

            plt.subplot(2,4,7)
            plt.imshow( (upright_geo[:, :, 1]+1.0)/2., cmap='gray') 

            plt.subplot(2,4,4)
            plt.imshow((cam_geo[:, :, 6:9] + 1.0)/2.) 

            plt.subplot(2,4,8)
            plt.imshow( (upright_geo+1.0)/2.) 


            plt.savefig(normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1])
            # combine_gt = np.concatenate((img_1, (cam_nu[:, :, 0:3] + 1.0)/2., (upright_nu[:, :, 0:3]+1.)/2.0, (cam_nu[:, :, 3:6] + 1.0)/2., (cam_nu[:, :, 3:6] + 1.0)/2.), axis=1) 

            # save_img = np.uint16( np.clip(np.round(65535.0 * combine_gt), 0., 65535.))
            # cv2.imwrite(normal_path.split('/')[-3] + '_' + normal_path.split('/')[-1], save_img[:, :, ::-1])

            print('InteriorNet train we are good')
            sys.exit()    

        final_img = torch.from_numpy(np.ascontiguousarray(img_1).transpose(2,0,1)).contiguous().float()
        targets_1['cam_geo'] = torch.from_numpy(np.ascontiguousarray(cam_geo).transpose(2,0,1)).contiguous().float()
        targets_1['upright_geo'] = torch.from_numpy(np.ascontiguousarray(upright_geo).transpose(2,0,1)).contiguous().float()
        
        targets_1['gt_mask'] = torch.from_numpy(np.ascontiguousarray(mask)).contiguous().float()
        # targets_1['atlanta_mask'] = torch.from_numpy(np.ascontiguousarray(atlanta_mask)).contiguous().float()

        targets_1['gt_rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
        targets_1['R_g_c'] = torch.from_numpy(np.ascontiguousarray(R_g_c)).contiguous().float()
        targets_1['gt_up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()

        targets_1['img_path'] = img_path
        targets_1['normal_path'] = normal_path
        targets_1['fx'] = fx
        targets_1['fy'] = fy

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


