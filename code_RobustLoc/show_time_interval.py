from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from transforms3d import quaternions
import numpy as np
import math
import time
import roma

import bisect
import csv
import numpy as np
import numpy.matlib as ml
# from .transform import *
from data.robotcar_sdk.python.transform import *
import os.path as osp

import os

def interpolate_ins_poses(ins_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from INS.

    Args:
        ins_path (str): path to file containing poses from INS.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(ins_path) as ins_file:
        ins_reader = csv.reader(ins_file)
        headers = next(ins_file)

        ins_timestamps = [0]
        abs_poses = [ml.identity(4)]

        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        for row in ins_reader:
            if row[1] != 'INS_SOLUTION_GOOD':
                continue
            timestamp = int(row[0])
            ins_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[5:7]+row[4:5]+row[12:]]
            abs_pose = build_se3_transform(xyzrpy)
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    ins_timestamps = ins_timestamps[1:]
    abs_poses = abs_poses[1:]

    return abs_poses







def main():
    seq_dir = '/data/abc/loc/RobotCar/loop/2014-06-23-15-36-04'
    timestamp = os.listdir('/data/abc/loc/RobotCar/loop/2014-06-23-15-36-04/stereo/centre_256/')
    timestamp = sorted(timestamp)
    timeinterval = []
    for i,each_timestamp in enumerate(timestamp):
        each_timestamp = each_timestamp.split('.png')[0]
        each_timestamp = int(each_timestamp)
        timestamp[i] = each_timestamp
        if i==0:
            timeinterval.append(0)
        else:
            timeinterval.append(timestamp[i]-timestamp[i-1])
    

    pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
    p = np.asarray(interpolate_ins_poses(pose_filename, timestamp, timestamp[0]))

    p = p[:,:3,:]
    p = p[:,:,-1]

    t = np.array(range(len(p)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,1], p[:,0], range(len(p)), s=1, alpha=1, label='targ_poses', marker='.')
    # ax.scatter(train_poses_924[:,1], train_poses_924[:,0], range(len(train_poses_924)),  s=1, alpha=1, label='train_poses_924', marker='.')
    # ax.scatter(train_poses_853[:,1], train_poses_853[:,0], range(len(train_poses_853)),  s=1, alpha=1, label='train_poses_853', marker='.')
    plt.legend()
    # plt.show()

    # plt.show()
    plt.savefig('1.png', bbox_inches="tight")

    return None


if __name__ == '__main__':
    main()