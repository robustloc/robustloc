import matplotlib.pyplot as plt
import numpy as np
from tools.utils import quaternion_angular_error
import os
plt.tight_layout()
# plt.switch_backend('QtAgg4')
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error


epoch = 235
# pred_poses = np.loadtxt('123/init_poses/1.txt')
# pred_poses = np.loadtxt('output_poses/ep{:s}_pred.txt'.format(str(epoch)))
# targ_poses = np.loadtxt('output_poses/ep{:s}_targ.txt'.format(str(epoch)))
pred_poses = np.loadtxt('123/ep{:s}_pred.txt'.format(str(epoch)))
targ_poses = np.loadtxt('123/ep{:s}_targ.txt'.format(str(epoch)))
# pred_poses = np.loadtxt('123/dropout_poses/ep{:s}_pred.txt'.format(str(epoch)))
# targ_poses = np.loadtxt('123/dropout_poses/ep{:s}_targ.txt'.format(str(epoch)))
train_poses_924 = np.loadtxt('data/poses_924.txt')
train_poses_853 = np.loadtxt('data/poses_853.txt')
# images_folder = '/data/abc/loc/RobotCar/loop/2014-06-23-15-36-04/stereo/centre_256/'
# images_list = os.listdir(images_folder)



def main1():
    t_loss = np.array([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    q_loss = np.array([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

    t_median_error = np.median(t_loss)
    q_median_error = np.median(q_loss)
    t_mean_error = np.mean(t_loss)
    q_mean_error = np.mean(q_loss)
    print('t median {:3.2f} m,  mean {:3.2f} m \nq median {:3.2f} degrees, mean {:3.2f} degree'\
                .format(t_median_error, t_mean_error, q_median_error, q_mean_error))

    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(targ_poses[:,1], targ_poses[:,0], range(len(targ_poses)), s=1, alpha=1, label='targ_poses', marker='.')
    ax.scatter(train_poses_924[:,1], train_poses_924[:,0], range(len(train_poses_924)),  s=1, alpha=1, label='train_poses_924', marker='.')
    ax.scatter(train_poses_853[:,1], train_poses_853[:,0], range(len(train_poses_853)),  s=1, alpha=1, label='train_poses_853', marker='.')
    plt.legend()
    # plt.show()

    # plt.show()
    plt.savefig('/data/abc/loc/cuda0_AtLoc/123/init_poses/1.png', bbox_inches="tight")




if __name__ == '__main__':
    main1()
    # main2()


