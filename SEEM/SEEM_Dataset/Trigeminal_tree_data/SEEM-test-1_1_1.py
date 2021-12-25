import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy
try:
    import ipdb
except:
    import pdb as ipdb
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--dset_type', default='test', type=str)
#导入math包
import math
#定义点的函数
class Point:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    def getx(self):
        return self.x
    def gety(self):
        return self.y 
#定义直线函数   
class Getlen:
    def __init__(self,p1,p2):
        self.x=p1.getx()-p2.getx()
        self.y=p1.gety()-p2.gety()
        #用math.sqrt（）求平方根
        self.len= math.sqrt((self.x**2)+(self.y**2))
    #定义得到直线长度的函数
    def getlen(self):
        return self.len


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    count_left=0
    count_mid=0
    count_right=0
    count=0
    with torch.no_grad():
        for batch in loader:

            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)


	    #---------------previous_tra,groundtruth_tra,predicted_tra--------------
            window_num = obs_traj.size(1)
            
            for i in range(window_num):
                print(i)
            #-----------------------------------------------------
                for ji in range(num_samples):

                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )

                    pred_traj_fake = relative_to_abs(               
                        pred_traj_fake_rel, obs_traj[-1]
                    )
	    #----------visualization-----------------------------------------------------------
                                 
                    predicted_tra=pred_traj_fake[:,i,:]
                    predicted_tra_x=predicted_tra[:,0]
                    predicted_tra_y=predicted_tra[:,1]
                    predicted_tra_x=predicted_tra_x.cpu()
                    predicted_tra_y=predicted_tra_y.cpu()
                    predicted_tra_x=predicted_tra_x.numpy()
                    predicted_tra_y=predicted_tra_y.numpy()

                    pred_x=predicted_tra_x[-1]
                    pred_y=predicted_tra_y[-1]
                    pred_last_pos=Point(pred_x,pred_y)
                    p1=Point(-8,16)
                    p2=Point(8,16)
                    left=Getlen(p1,pred_last_pos)
                    left=left.getlen()
                    right=Getlen(p2,pred_last_pos)
                    right=right.getlen()
                    #ipdb.set_trace()
                    if -1.5 < pred_x < 1.8:
                        count_mid+=1
                    else:
                        if left >= right:
                            count_right+=1
                        else:
                            count_left+=1

            #----------------------------------------------------------------------------------
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))
            
            

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            print("第%d轮batch结束" % count)
            count+=1
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        
        return ade, fde, count_left, count_mid, count_right


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)

        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde, count_left, count_mid, count_right= evaluate(_args, loader, generator, args.num_samples)
        print(path,'\n','ADE: {:.2f}, FDE: {:.2f}'.format(ade, fde))
        print("count_left: ",count_left," count_mid: ",count_mid,"  count_right: ",count_right)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
