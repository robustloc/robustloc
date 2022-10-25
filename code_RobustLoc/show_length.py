import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


train_poses_924 = np.loadtxt('data/poses_924.txt') # 924-3164  853-3040
train_poses_853 = np.loadtxt('data/poses_853.txt') # 924-3164  853-3040
train_poses_targ = np.loadtxt('data/ep235_targ.txt')  # 3438
# train_poses_pred = np.loadtxt('data/ep235_pred.txt') 

def show_length():

    length_924 = []
    speed_924 = []
    for i, _ in enumerate(train_poses_924):
        if i==0:
            init_translation = np.linalg.norm(train_poses_924[0,:3]-train_poses_targ[0,:3])
            length_924.append(init_translation)
            speed_924.append(0)
        elif i>0:
            translation_targ = np.linalg.norm(train_poses_924[i,:3]-train_poses_924[i-1,:3])
            length_924.append(translation_targ+length_924[i-1])
            speed_924.append(translation_targ)


    length_853 = []
    speed_853 = []
    for i, _ in enumerate(train_poses_853):
        if i==0:
            init_translation = np.linalg.norm(train_poses_853[0,:3]-train_poses_targ[0,:3])
            length_853.append(init_translation)
            speed_853.append(0)
        elif i>0:
            translation_targ = np.linalg.norm(train_poses_853[i,:3]-train_poses_853[i-1,:3])
            length_853.append(translation_targ+length_853[i-1])
            speed_853.append(translation_targ)


    length_targ = []
    length_pred = []
    speed_targ = []
    for i, _ in enumerate(train_poses_targ):
        if i==0:
            length_targ.append(0)
            # length_pred.append(0)
            speed_targ.append(0)
        elif i>0:
            translation_targ = np.linalg.norm(train_poses_targ[i,:3]-train_poses_targ[i-1,:3])
            # translation_pred = np.linalg.norm(train_poses_pred[i,:3]-train_poses_pred[i-1,:3])
            length_targ.append(translation_targ+length_targ[i-1])
            # length_pred.append(translation_pred+length_pred[i-1])
            speed_targ.append(translation_targ)


    # ---- speed augmentation 1
    length_924_aug1 = []
    speed_924_aug1 = copy.deepcopy(speed_924)
    for i, _ in enumerate(speed_924_aug1):
        speed_924_aug1[i] += (random.random()-0.4)*0.2
        if i==0:
            length_924_aug1.append(0)
        else:
            length_924_aug1.append(length_924_aug1[i-1]+speed_924_aug1[i])




    # ---- speed augmentation 2
    length_924_aug2 = []
    speed_924_aug2 = copy.deepcopy(speed_924)
    pop_ids = np.random.randint(0, len(speed_924_aug2), 100)
    speed_924_aug2 = np.array(speed_924_aug2)
    speed_924_aug2 = np.delete(speed_924_aug2, pop_ids)
    for i, _ in enumerate(speed_924_aug2):
        if i==0:
            length_924_aug2.append(0)
        else:
            length_924_aug2.append(length_924_aug2[i-1]+speed_924_aug2[i])
    




    fig = plt.figure(figsize=[20,16],dpi=100)
    plt.plot(length_924, label='924',)
    plt.plot(length_853, label='853')
    plt.plot(length_targ, '+', label='targ')
    plt.plot(length_924_aug1, label='length_924_aug1')
    plt.plot(length_924_aug2, label='length_924_aug2')
    # plt.plot(length_pred, label='pred')
    plt.legend()
    plt.savefig('length_all.png',bbox_inches='tight')



    fig = plt.figure(figsize=[20,16],dpi=100,)
    plt.plot(speed_924, '.', label='speed_924')
    plt.plot(speed_853, '.', label='speed_853')
    plt.plot(speed_targ, '.', label='speed_targ')
    plt.plot(speed_924_aug1, '.', label='speed_924_aug1')
    plt.plot(speed_924_aug2, '.', label='speed_924_aug2')
    plt.legend()
    plt.savefig('speed_all.png', bbox_inches='tight')
1







def show_angle():
    angle_924 = []
    angle_3d_924 = []
    # speed_924 = []
    for i, _ in enumerate(train_poses_924):

        rotation = np.linalg.norm(train_poses_924[i,-3:])
        rotation_3d = train_poses_924[i,-3:]

        angle_924.append(rotation)
        angle_3d_924.append(rotation_3d)
        # speed_924.append(rotation)



    fig = plt.figure(figsize=[20,16],dpi=100,)
    plt.plot(angle_924, label='924')
    # plt.plot(length_853, label='853')
    # plt.plot(length_targ, label='targ')
    # plt.plot(length_pred, label='pred')
    plt.legend()
    plt.savefig('angle_all.png',bbox_inches='tight')



    # fig = plt.figure(figsize=[20,16],dpi=100)
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot(angle_924, label='924')
    # # plt.plot(length_853, label='853')
    # # plt.plot(length_targ, label='targ')
    # # plt.plot(length_pred, label='pred')
    # plt.legend()
    # plt.savefig('angle_all.png',bbox_inches='tight')





def show_angle_3d():
    angle_3d_924 = []
    for i, _ in enumerate(train_poses_924):

        rotation = np.linalg.norm(train_poses_924[i,-3:])
        rotation_3d = train_poses_924[i,-3:]

        angle_3d_924.append(rotation_3d)
        # speed_924.append(rotation)





    fig = plt.figure(figsize=[20,16],dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(train_poses_924[:,-4], 
        train_poses_924[:,-3],
        train_poses_924[:,-2],
        label='924')
    # plt.plot(length_853, label='853')
    # plt.plot(length_targ, label='targ')
    # plt.plot(length_pred, label='pred')
    plt.legend()
    plt.show()
    plt.savefig('angle_3d.png',bbox_inches='tight')




if __name__ == '__main__':
    show_length()
    # show_angle()
    # show_angle_3d()