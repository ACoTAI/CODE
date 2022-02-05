import argparse
import os
import torch
import sys
sys.path.append('..')
import time
import gc
import logging
import  ipdb
import numpy as np
from scipy.stats import gaussian_kde

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from attrdict import AttrDict
from sgan.data.loader import data_loader
from sgan.models import MALA_corrected_sampler, get_sgld_proposal
from sgan.models import TrajectoryGeneratorSO, TrajectoryGeneratorST, TrajectoryDiscriminator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = 12
    num_batches = 1


    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll

def get_generatorSO(checkpoint):
    args = AttrDict(checkpoint['args'])
    generatorSO = TrajectoryGeneratorSO(
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
    generatorSO.load_state_dict(checkpoint['gso_state'])
    generatorSO.cuda()
    generatorSO.train()
    return generatorSO

def get_generatorST(checkpoint):
    args = AttrDict(checkpoint['args'])
    generatorST = TrajectoryGeneratorST(
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
    generatorST.load_state_dict(checkpoint['gst_state'])
    generatorST.cuda()
    generatorST.train()
    return generatorST

def get_dicriminator(checkpoint):
    args = AttrDict(checkpoint['args'])
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)
    discriminator.load_state_dict(checkpoint['d_state'])
    discriminator.cuda()
    discriminator.train()
    return discriminator

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

#TODO:generatorSO, generatorST, discriminator
def evaluate(args, loader, generatorSO, generatorST, discriminator, num_samples):
    ade_outer, fde_outer, kde_outer = [], [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                noise_input, noise_shape = generatorSO(obs_traj, obs_traj_rel, seq_start_end)
                with torch.enable_grad():
                    z_noise = MALA_corrected_sampler(generatorST, discriminator, args, noise_shape, noise_input, seq_start_end, obs_traj, obs_traj_rel)
                decoder_h = torch.cat([noise_input, z_noise], dim=1)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                generator_out = generatorST(decoder_h, seq_start_end, obs_traj, obs_traj_rel)
                pred_traj_fake_rel = generator_out
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)
            kde_sum = compute_kde_nll(pred_traj_fake, pred_traj_gt)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            kde_outer.append(kde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        kde_nll = sum(kde_outer) / len(kde_outer)
        return ade, fde, kde_nll


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
        #TODO:gso&gst
        generatorSO = get_generatorSO(checkpoint)
        generatorST = get_generatorST(checkpoint)
        discriminator = get_dicriminator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde, KDE_NLL = evaluate(_args, loader, generatorSO, generatorST, discriminator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, KDE_NLL: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde, KDE_NLL))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
