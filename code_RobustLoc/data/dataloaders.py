import os
import random
import re
from xml.dom import IndexSizeErr
import cv2
import torch
import numpy as np
import pickle
import os.path as osp

from data.robotcar_sdk.python.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from data.robotcar_sdk.python.camera_model import CameraModel
from data.robotcar_sdk.python.image import load_image as robotcar_loader
from tools.utils import process_poses, calc_vos_simple, load_image
from torch.utils import data
from functools import partial
from tools.options import Options
from PIL import Image
from tools.utils import qexp
from torchvision import transforms
import torchvision.transforms.functional as TVF
from tools.utils import set_seed
set_seed(7)
opt = Options().parse()


class RobotCar(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, real=False, 
        skip_images=False, seed=7, undistort=False, vo_lib='stereo', subseq_length=10):
        np.random.seed(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undistort
        self.subseq_length = subseq_length

        self.train = train

        # directories
        data_dir = osp.join(data_path, 'RobotCar', scene)


        # ---- train test split
        if scene=='loop':
            if train:
                seqs = [
                    # '2014-06-23-15-36-04',
                    # '2014-06-23-15-41-25',
                    '2014-06-26-08-53-56',
                    '2014-06-26-09-24-58', 
                ]
            elif not train:
                seqs = [
                    '2014-06-23-15-36-04',
                    '2014-06-23-15-41-25',
                    # '2014-06-26-08-53-56',
                    # '2014-06-26-09-24-58', 
                ]



        elif scene=='full':
            if train:
                seqs = [
                    '2014-11-28-12-07-13',
                    '2014-12-02-15-30-08',
                ]
            elif not train:
                seqs = [
                    '2014-12-09-13-21-02',
                ]

        # if not opt.x4fps:
        #     self.num_images_seq = {
        #         '2014-06-23-15-36-04':3438,
        #         '2014-06-23-15-41-25':3356-13, # original
        #         '2014-06-26-08-53-56':3040,
        #         '2014-06-26-09-24-58':3164, 
        #     }
        # elif opt.x4fps:
        #     self.num_images_seq = {
        #         '2014-06-23-15-36-04':13749,
        #         '2014-06-23-15-41-25':13369-0, # original
        #         '2014-06-26-08-53-56':12157,
        #         '2014-06-26-09-24-58':12653, 
        #     }
            

        ps = {}
        ts = {}
        self.img_paths = []
        self.depth_paths = []
        # self.semantic_mask_paths = []
        # self.semantic_mask_psp_paths = []

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)

            # ---- read from folder 
            if self.train:
                ts[seq] = [int(image_name.split('.png')[0]) for image_name in os.listdir(osp.join(seq_dir, 'stereo', 'centre'))] 
            elif not self.train:
                ts[seq] = [int(image_name.split('.png')[0]) for image_name in os.listdir(osp.join(seq_dir, 'stereo', 'centre'))] 



            ts[seq] = sorted(ts[seq]) 
            if seq=='2014-06-23-15-41-25':
                if not opt.x4fps:
                    ts[seq] = ts[seq][:-13]
                    assert len(ts[seq]) == 3343
                elif opt.x4fps:
                    assert len(ts[seq]) == 13369





            
            if self.train:
                self.img_paths.extend(
                    [osp.join(seq_dir, 'stereo', 'centre_{:s}'.format(str(opt.cropsize)), '{:d}.png'.format(t)) for t in ts[seq]])

                # self.depth_paths.extend(
                #     [osp.join(seq_dir, 'stereo', 'depth_{:s}'.format(str(opt.cropsize)), '{:d}_disp.png'.format(t)) for t in ts[seq]])


            elif not self.train:
                if opt.attack_name is None:
                    self.img_paths.extend(
                        [osp.join(seq_dir, 'stereo', 'centre_{:s}'.format(str(opt.cropsize)), '{:d}.png'.format(t)) for t in ts[seq]])
                elif opt.attack_name=='snow1_spatter3':
                    self.img_paths.extend(
                        [osp.join(seq_dir, 'stereo', 'centre_{:s}_{:s}'.format(
                            str(opt.cropsize), opt.attack_name), '{:d}.png'.format(t)) for t in ts[seq]])
                # self.depth_paths.extend(
                #     [osp.join(seq_dir, 'stereo', 'depth_{:s}'.format(str(opt.cropsize)), '{:d}_disp.png'.format(t)) for t in ts[seq]])

 





        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        mean_t, std_t = np.loadtxt(pose_stats_filename)

        if self.train: # train
            if scene=='loop':
                ps = {
                    '2014-06-23-15-36-04':np.loadtxt('data/RobotCar_poses/2014-06-23-15-36-04_tR.txt'),
                    '2014-06-23-15-41-25':np.loadtxt('data/RobotCar_poses/2014-06-23-15-41-25_tR.txt'),
                    '2014-06-26-08-53-56':np.loadtxt('data/RobotCar_poses/2014-06-26-08-53-56_tR.txt'),
                    '2014-06-26-09-24-58':np.loadtxt('data/RobotCar_poses/2014-06-26-09-24-58_tR.txt'), 
                }
            elif scene=='full':
                ps = {
                    '2014-11-28-12-07-13':np.loadtxt('data/RobotCar_poses/2014-11-28-12-07-13_tR.txt'),
                    '2014-12-02-15-30-08':np.loadtxt('data/RobotCar_poses/2014-12-02-15-30-08_tR.txt'),
                    '2014-12-09-13-21-02':np.loadtxt('data/RobotCar_poses/2014-12-09-13-21-02_tR.txt'),
                }


        elif not self.train: # test
            if scene=='loop':
                ps = {
                    '2014-06-23-15-36-04':np.loadtxt('data/RobotCar_poses/2014-06-23-15-36-04_tR.txt'),
                    '2014-06-23-15-41-25':np.loadtxt('data/RobotCar_poses/2014-06-23-15-41-25_tR.txt'),
                    '2014-06-26-08-53-56':np.loadtxt('data/RobotCar_poses/2014-06-26-08-53-56_tR.txt'),
                    '2014-06-26-09-24-58':np.loadtxt('data/RobotCar_poses/2014-06-26-09-24-58_tR.txt'), 
                }
            elif scene=='full':
                ps = {
                    '2014-11-28-12-07-13':np.loadtxt('data/RobotCar_poses/2014-11-28-12-07-13_tR.txt'),
                    '2014-12-02-15-30-08':np.loadtxt('data/RobotCar_poses/2014-12-02-15-30-08_tR.txt'),
                    '2014-12-09-13-21-02':np.loadtxt('data/RobotCar_poses/2014-12-09-13-21-02_tR.txt'),
                }





        self.samples_length = []
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                              align_R=np.eye(3), align_t=np.zeros(3),
                              align_s=1)
            self.poses = np.vstack((self.poses, pss))
            self.samples_length.append(len(pss))


        
        1


    
    def __len__(self):
        # if self.train:
        #     self.samples_length_924 = 3164
        #     self.samples_length_853 = 3040
        #     return self.samples_length_924 + self.samples_length_853
        # else:
        #     self.samples_length_test = 3438
        #     return self.samples_length_test
        # l = 0
        # for 
        return len(self.poses)
        


    # ---- index is pure Int, not array/tensor
    def get_indices(self, index): 
        if self.train:
            if index >= self.samples_length[0]:
                indices_pool = np.array(range(self.samples_length[0], self.__len__()))
            else:
                indices_pool = np.array(range(0, self.samples_length[0]))
        
        elif not self.train:
            # indices_pool = np.array(range(0, self.__len__()))
            if index >= self.samples_length[0]:
                indices_pool = np.array(range(self.samples_length[0], self.__len__()))
            else:
                indices_pool = np.array(range(0, self.samples_length[0]))

        # indices_pool = indices_pool-index

        # output_indices = np.array(range(-(opt.subseq_length//2), (opt.subseq_length//2)+1)) 
        output_indices = np.arange(-(opt.subseq_length//2)*opt.skip, (opt.subseq_length//2)*opt.skip+1, opt.skip) + index

        # ---- random shift
        if self.train:
            for i_index, each_index in enumerate(output_indices):
                if i_index==len(output_indices)//2:
                    continue
                else:
                    if random.random()<opt.shift_prob:
                        index_shift = random.randint(-opt.shift_range, opt.shift_range)
                        index_shifted = each_index + index_shift
                        output_indices[i_index] = index_shifted

            assert output_indices[len(output_indices)//2]==index

        # ---- check if outrange
        for i_index, each_index in enumerate(output_indices):
            if each_index > indices_pool[-1]:
                output_indices[i_index] = indices_pool[-1]
            elif each_index < indices_pool[0]:
                output_indices[i_index] = indices_pool[0]

        return output_indices
        

    def __getitem__(self, index):


        image_name = self.img_paths[index].split('/')[-1]
        image_name_pure = image_name.split('.')[0]
        

        indices_list = self.get_indices(index)
        

        timestamps_list = []
        for each_index in indices_list:
            image_name = self.img_paths[each_index].split('/')[-1]
            image_name_pure = image_name.split('.')[0]
            each_timestamp = int(image_name_pure)
            each_timestamp = torch.tensor(each_timestamp)
            timestamps_list.append(each_timestamp)
        timestamps_list = torch.stack(timestamps_list, dim=0)



        if self.train:
            imgs = []
            depths = []

            for index in indices_list:
                img = load_image(self.img_paths[index])
                # depth = Image.open(self.depth_paths[index]).convert('L')



                # ---- crop
                if opt.random_crop:
                    i, j, th, tw = transforms.RandomCrop(size=opt.cropsize).get_params(
                        img, output_size=[opt.cropsize, opt.cropsize])
                    img = TVF.crop(img, i, j, th, tw)
                    # depth = TVF.crop(depth, i, j, th, tw)
                else:
                    img = transforms.CenterCrop(opt.cropsize)(img)
                    # depth = transforms.CenterCrop(opt.cropsize)(depth)
                

                
                imgs.append(img)
                # depths.append(depth)
            


        elif not self.train:
            imgs = []
            # depths = []
            for index in indices_list:
                img = load_image(self.img_paths[index])
                # depth = Image.open(self.depth_paths[index]).convert('L')
                # ---- crop
                img = transforms.CenterCrop(opt.cropsize)(img)
                # depth = transforms.CenterCrop(opt.cropsize)(depth)



                imgs.append(img)
                # depths.append(depth)


        
        poses = []
        for index in indices_list:
            pose = np.float32(self.poses[index])
            poses.append(pose)



        # ---- transform pose and img
        for index_in_subseq, _ in enumerate(poses):
            if self.target_transform is not None:
                poses[index_in_subseq] = self.target_transform(poses[index_in_subseq])
            if self.transform is not None:
                imgs[index_in_subseq] = self.transform(imgs[index_in_subseq])
                # depths[index_in_subseq] = transforms.ToTensor()(depths[index_in_subseq])
                # depths[index_in_subseq] = depths[index_in_subseq] - 0.5
                # if self.train:
                #     if random.random()<opt.depth_scale_prob:
                #         rand_scale = (random.random()-0.5)*2*0.1 + 1
                #         rand_shift = (random.random()-0.5)*2*0.1
                #         depths[index_in_subseq] *= rand_scale
                #         depths[index_in_subseq] += rand_shift


        # ---- return
        if self.train:
            imgs = torch.stack(imgs)
            poses = torch.stack(poses)
            # depths = torch.stack(depths)
            return imgs, poses, timestamps_list

        else:
            imgs = torch.stack(imgs)
            poses = torch.stack(poses)
            # depths = torch.stack(depths)
            return imgs, poses, image_name_pure, timestamps_list


