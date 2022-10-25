import matplotlib.pyplot as plt
import numpy as np
from tools.utils import quaternion_angular_error
import os
# plt.switch_backend('QtAgg4')
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error


epoch = 51
# pred_poses = np.loadtxt('123/init_poses/1.txt')
# pred_poses = np.loadtxt('output_poses/ep{:s}_pred.txt'.format(str(epoch)))
# targ_poses = np.loadtxt('output_poses/ep{:s}_targ.txt'.format(str(epoch)))
pred_poses = np.loadtxt('output_poses/ep{:s}_pred.txt'.format(str(epoch)))
targ_poses = np.loadtxt('output_poses/ep{:s}_targ.txt'.format(str(epoch)))
# pred_poses = np.loadtxt('123/dropout_poses/ep{:s}_pred.txt'.format(str(epoch)))
# targ_poses = np.loadtxt('123/dropout_poses/ep{:s}_targ.txt'.format(str(epoch)))
train_poses_924 = np.loadtxt('data/RobotCar_poses/2014-06-26-09-24-58_tR.txt')
train_poses_853 = np.loadtxt('data/RobotCar_poses/2014-06-26-08-53-56_tR.txt')


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


    
    norm = plt.Normalize(t_loss.min(), t_loss.max())
    norm_y = norm(t_loss)

    # plt.scatter(targ_poses[:, 1], targ_poses[:, 0], cmap='jet',linewidths=0.2)
    # plt.plot(pred_poses[:, 1], pred_poses[:, 0], color='green', linewidth=1, label='pred_poses')
    # plt.plot(targ_poses[:, 1], targ_poses[:, 0], color='orange', linewidth=1, label='targ_poses')
    # plt.plot(train_poses_924[:, 1], train_poses_924[:, 0], color='blue', linewidth=5, label='train_poses_924')
    # plt.plot(train_poses_853[:, 1], train_poses_853[:, 0], color='yellow', linewidth=3, label='train_poses_853')
    plt.plot(pred_poses[:, 1], pred_poses[:, 0], color='red', linewidth=0.5)
    plt.scatter(targ_poses[:, 1], targ_poses[:, 0], c=norm_y, cmap='jet',linewidths=1)
    plt.plot(targ_poses[0, 1], targ_poses[0, 0], 'y*', markersize=15)


    
    # plt.plot(pred_poses[:, 1], pred_poses[:, 0], color='red', linewidth=0.5)
    # plt.legend()

    plt.show()
    plt.savefig('1.png')


# ---- 剔除一些离群点
def main2():
    t_loss = []
    q_loss = []
    pred_poses_refined = np.empty([0,7])
    targ_poses_refined = np.empty([0,7])

    for i, (pred_pose, targ_pose) in enumerate(zip(pred_poses, targ_poses)):
        t_loss_each = t_criterion(pred_pose[:3], targ_pose[:3])
        q_loss_each = q_criterion(pred_pose[3:], targ_pose[3:])
    
        if t_loss_each<15 and q_loss_each<3:
            t_loss.append(t_loss_each)
            q_loss.append(q_loss_each)
            pred_poses_refined = np.vstack([pred_poses_refined, pred_pose[np.newaxis, :]])
            targ_poses_refined = np.vstack([targ_poses_refined, targ_pose[np.newaxis, :]])




    # t_loss = np.array([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    # q_loss = np.array([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

    t_median_error = np.median(t_loss)
    q_median_error = np.median(q_loss)
    t_mean_error = np.mean(t_loss)
    q_mean_error = np.mean(q_loss)
    print('t median {:3.2f} m,  mean {:3.2f} m \nq median {:3.2f} degrees, mean {:3.2f} degree'\
                .format(t_median_error, t_mean_error, q_median_error, q_mean_error))

    plt.scatter(targ_poses_refined[:, 1], targ_poses_refined[:, 0], cmap='jet',linewidths=0.1)
    plt.plot(pred_poses_refined[:, 1], pred_poses_refined[:, 0], color='red', linewidth=0.5)
    plt.show()
    plt.savefig('2.png')







if __name__ == '__main__':
    # main1()
    main2()


