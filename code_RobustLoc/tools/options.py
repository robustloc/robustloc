import argparse
import os
# from tools import utils
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        # ---- machine related
        self.parser.add_argument('--data_dir', type=str, default='/data/abc/loc/')
        self.parser.add_argument('--cuda', type=float, default=3)
        self.parser.add_argument('--nThreads', type=int, default=6)
        self.parser.add_argument('--num_tries', type=int, default=1)
        self.parser.add_argument('--resume_epoch', type=int, default=-1)
        self.parser.add_argument('--eval_epoch', type=int, default=100)
        self.parser.add_argument('--eval_interval', type=int, default=1)
        self.parser.add_argument('--semantic_aug', type=bool, default=False)
        self.parser.add_argument('--x4fps', type=bool, default=False)
        self.parser.add_argument('--ode_method', type=str, default='euler')
        self.parser.add_argument('--scene', type=str, default='loop')
        self.parser.add_argument('--subseq_length', type=int, default=9)
        self.parser.add_argument('--skip', type=int, default=10)
        self.parser.add_argument('--shift_range', type=int, default=7)
        self.parser.add_argument('--shift_prob', type=float, default=0.0)
        self.parser.add_argument('--depth_scale_prob', type=float, default=0.0)


        self.parser.add_argument('--num_gattlayers', type=int, default=3)
        self.parser.add_argument('--branchres', type=bool, default=False)
        self.parser.add_argument('--sumout', type=bool, default=False)
        self.parser.add_argument('--droppath', type=float, default=0)
        self.parser.add_argument('--odefc', type=int, default=1)
        self.parser.add_argument('--gattnorm', type=str, default='ln')
        self.parser.add_argument('--gattactivation', type=str, default='relu')
        self.parser.add_argument('--attack_name', type=str, default=None)




        self.parser.add_argument('--cropsize', type=int, default=128)
        self.parser.add_argument('--random_crop', type=bool, default=True)
        self.parser.add_argument('--resize_range', type=tuple, default=(1,1))
        self.parser.add_argument('--thd', type=float, default=0.5)
        self.parser.add_argument('--dilate', type=str, default='7*7', help='(x,y)')
        self.parser.add_argument('--aug_rate', type=float, default=0.5)
        self.parser.add_argument('--aug_mode', type=str, default='add')
        self.parser.add_argument('--mask_class', type=str, default='car')
        self.parser.add_argument('--mask_flip_rate', type=float, default=0.5)
        self.parser.add_argument('--color_jitter', type=float, default=1)
        self.parser.add_argument('--mask_percent', type=float, default=0.)
        
        
        self.parser.add_argument('--num_atts', type=int, default=1)
        self.parser.add_argument('--layer_att', type=bool, default=False)
        self.parser.add_argument('--fc_dim', type=int, default=2048)
        # self.parser.add_argument('--conv1_size', type=int, default=7)
        self.parser.add_argument('--last_conv_dim', type=int, default=None)

        # ---- gnn
        self.parser.add_argument('--batchsize', type=int, default=64)
        # self.parser.add_argument('--neighbor_fc_dim', type=int, default=256)
        self.parser.add_argument('--ode_hidden_dim', type=int, default=256)





        self.parser.add_argument('--save_out', type=bool, default=False)
        self.parser.add_argument('--save_fig', type=bool, default=True)
        self.parser.add_argument('--save_image', type=bool, default=False)

        # base options
        self.parser.add_argument('--print_freq', type=int, default=20)
        self.parser.add_argument('--gpus', type=str, default='-1')
        self.parser.add_argument('--dataset', type=str, default='RobotCar')
        self.parser.add_argument('--model', type=str, default='AtLocPlus')
        self.parser.add_argument('--seed', type=int, default=7)
        self.parser.add_argument('--lstm', type=bool, default=False)
        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')
        self.parser.add_argument('--variable_skip', type=bool, default=False)
        self.parser.add_argument('--real', type=bool, default=False)
        self.parser.add_argument('--steps', type=int, default=3)
        self.parser.add_argument('--val', type=bool, default=True)


        # train options
        self.parser.add_argument('--epochs', type=int, default=300)
        self.parser.add_argument('--beta', type=float, default=-3.0)
        self.parser.add_argument('--gamma', type=float, default=-3.0, help='only for AtLoc+ (-3.0)')
        self.parser.add_argument('--train_dropout', type=float, default=0.5)
        self.parser.add_argument('--val_freq', type=int, default=5)
        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')
        self.parser.add_argument('--lr', type=float, default=2e-4)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)

        



        # test options
        self.parser.add_argument('--test_dropout', type=float, default=0.0)
        self.parser.add_argument('--weights', type=str, default='epoch_300.pth.tar')
        self.parser.add_argument('--save_freq', type=int, default=5)


    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpus.split(',')
        self.opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpus.append(id)

        # set gpu ids
        if len(self.opt.gpus) > 0:
            torch.cuda.set_device(self.opt.gpus[0])

        args = vars(self.opt)
        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ---------------')

        # save to the disk
        self.opt.exp_name = '{:s}_{:s}_{:s}_{:s}'.format(self.opt.dataset, self.opt.scene, self.opt.model, str(self.opt.lstm))
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
        self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])

        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda)

        return self.opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)