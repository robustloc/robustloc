import os
import numpy as np
import os.path as osp

from data.robotcar_sdk.python.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from transforms3d import quaternions
import copy



def main(robotcar_folder, x4fps, loop):

    if x4fps:
        x4fps_suffix = 'x4fps'
    else:
        x4fps_suffix = ''

    if loop:
        loop_suffix = 'loop'
    else:
        loop_suffix = 'full'

    seq_dir_list = []

    if loop:
        seq_dir_list.append('2014-06-23-15-36-04')
        seq_dir_list.append('2014-06-23-15-41-25')
        seq_dir_list.append('2014-06-26-08-53-56')
        seq_dir_list.append('2014-06-26-09-24-58')
    elif not loop:
        seq_dir_list.append('2014-11-28-12-07-13')
        seq_dir_list.append('2014-12-02-15-30-08')
        seq_dir_list.append('2014-12-09-13-21-02')



    for seq_dir in seq_dir_list:
        ts = []
        # images_folder = robotcar_folder + 'loop/' + seq_dir + '/stereo/centre_128{:s}/'.format(x4fps_suffix)
        images_folder = robotcar_folder + loop_suffix + '/' + seq_dir + '/stereo/centre_128{:s}/'.format(x4fps_suffix)
        images_list = os.listdir(images_folder)
        images_list = sorted(images_list)
        for image_name in images_list:
            timestamp = int(image_name.split('.png')[0])
            ts.append(timestamp)

        1



        pose_filename = 'data/RobotCar/{:s}/'.format(loop_suffix) +seq_dir+'/gps/ins.csv'

        p = np.asarray(interpolate_ins_poses(pose_filename, copy.copy(ts), ts[0]))

        t = p[:,:3,-1]
        R = p[:,:3,:3]
        q = []
        tR = p[:,:3,:4].reshape([-1,12])

        if seq_dir=='2014-06-23-15-41-25' and (not x4fps):
            t = t[:-13]
            R = R[:-13]
            tR = tR[:-13]


        for each_R in R:
            each_q = quaternions.mat2quat(each_R)
            q.append(each_q)

        q = np.array(q)

        tq = np.hstack([t,q])

        # np.savetxt('data/RobotCar_poses/{:s}_tq{:s}.txt'.format(seq_dir, x4fps_suffix), tq, fmt='%8.8f')
        np.savetxt('data/RobotCar_poses/{:s}_tR{:s}.txt'.format(seq_dir, x4fps_suffix), tR, fmt='%8.8f')

        1




if __name__ == '__main__':
    main('/data/sijie/loc/RobotCar/', x4fps=False, loop=True)