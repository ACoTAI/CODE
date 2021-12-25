import torch
import argparse
import os
import time
import sys
import ipdb
from attrdict import AttrDict
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from sgan.models import MALA_corrected_sampler, get_sgld_proposal
from sgan.models import TrajectoryGeneratorSO, TrajectoryGeneratorST, TrajectoryDiscriminator
from evaluate_model import get_generatorSO, get_generatorST ,get_dicriminator
from sgan.data.loader import data_loader
from sgan.utils import get_dset_path, relative_to_abs
from tensorboardX import SummaryWriter

localtime = time.asctime(time.localtime(time.time()))
localtime = localtime.replace(':','_').replace(' ','_')

parser = argparse.ArgumentParser()
parser.add_argument('--pt_name', type=str)
parser.add_argument('--image_name', default='image', type=str)
parser.add_argument('--net_name', default='net', type=str)
# parser.add_argument('--dest_path', default='E:\\GithubWorkSpace\\FinalDesign\\social_GAN\\sgan-master\\datasets\\eth\\test', type=str)
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--dset_type', default='test', type=str)

def plot_G_losses(args, checkpoint_Gloss):
    for loss_type in checkpoint_Gloss:
        with SummaryWriter(args.pt_name+'/'+loss_type, loss_type) as writer:
            for epoch in range(len(checkpoint_Gloss[loss_type])):
                writer.add_scalar(loss_type,checkpoint_Gloss[loss_type][epoch],epoch)

def plot_D_losses(args, checkpoint_Dloss):
    for loss_type in checkpoint_Dloss:
        with SummaryWriter(args.pt_name+'/'+loss_type, loss_type) as writer:
            for epoch in range(len(checkpoint_Dloss[loss_type])):
                writer.add_scalar(loss_type,checkpoint_Dloss[loss_type][epoch],epoch)

def plot_net_structure(args, generatorSO, generatorST, discriminator, obs_traj, obs_traj_rel, seq_start_end, decoder_h, traj_fake, traj_fake_rel):
    with SummaryWriter(args.pt_name+'/'+'discriminator', args.net_name) as writer:
        # ipdb.set_trace()
        writer.add_graph(discriminator, (traj_fake, traj_fake_rel, seq_start_end))
    # with SummaryWriter(args.pt_name+'/'+'generatorSO', args.net_name) as writer:
    #     ipdb.set_trace()
    #     writer.add_graph(generatorSO, (traj_fake, traj_fake_rel, seq_start_end),verbose=True)
    # with SummaryWriter(args.pt_name+'/'+'generatorST', args.net_name) as writer:
    #     ipdb.set_trace()
    #     writer.add_graph(generatorST, (decoder_h, seq_start_end, obs_traj, obs_traj_rel))
        
def main(args):



    checkpoint = torch.load(args.pt_name+'.pt')
    checkpoint_Gloss = checkpoint['G_losses']
    checkpoint_Dloss = checkpoint['D_losses']
    _args = AttrDict(checkpoint['args'])

    generatorSO = get_generatorSO(checkpoint)
    generatorST = get_generatorST(checkpoint)
    discriminator = get_dicriminator(checkpoint)

    path = get_dset_path(_args.dataset_name, args.dset_type)
    _, loader = data_loader(_args, path)

    batch_final = ()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        batch_final = batch
    
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
    #ipdb.set_trace()
    noise_input, noise_shape = generatorSO(obs_traj, obs_traj_rel, seq_start_end)
    z_noise = MALA_corrected_sampler(generatorST, discriminator, _args, noise_shape, noise_input, seq_start_end, obs_traj, obs_traj_rel)
    decoder_h = torch.cat([noise_input, z_noise], dim=1)
    decoder_h = torch.unsqueeze(decoder_h, 0)
    generator_out = generatorST(decoder_h, seq_start_end, obs_traj, obs_traj_rel)
    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    
    plot_G_losses(args, checkpoint_Gloss)
    plot_D_losses(args, checkpoint_Dloss)
    plot_net_structure(args, generatorSO, generatorST, discriminator, obs_traj, obs_traj_rel, seq_start_end, decoder_h, traj_fake, traj_fake_rel)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)