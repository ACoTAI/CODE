import logging
import os
import math

import numpy as np

import torch


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

data_dir='D:\\train'
obs_len=8
pred_len=12
skip=1
threshold=0.002
min_ped=1
delim='\t'
seq_len = obs_len + pred_len
all_files = os.listdir(data_dir)
all_files = [os.path.join(data_dir, _path) for _path in all_files]
num_peds_in_seq = []
seq_list = []
seq_list_rel = []
loss_mask_list = []
non_linear_ped = []
for path in all_files:
    data = read_file(path, delim)
    frames = np.unique(data[:, 0]).tolist() #np.unique去掉重复值，保留不同值
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])
    num_sequences = int(
        math.ceil((len(frames) - seq_len + 1) / skip))
    for idx in range(0, num_sequences * skip + 1, skip):
        curr_seq_data = np.concatenate(
            frame_data[idx:idx + seq_len], axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                 seq_len))
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
        curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                   seq_len))
        num_peds_considered = 0
        _non_linear_ped = []
        for _, ped_id in enumerate(peds_in_curr_seq):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                         ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
            if pad_end - pad_front != seq_len:
                continue
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
            curr_ped_seq = curr_ped_seq
            # Make coordinates relative
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            _idx = num_peds_considered
            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
            curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
            # Linear vs Non-Linear Trajectory
            _non_linear_ped.append(
                poly_fit(curr_ped_seq, pred_len, threshold))
            curr_loss_mask[_idx, pad_front:pad_end] = 1
            num_peds_considered += 1
        if num_peds_considered > min_ped:
            non_linear_ped += _non_linear_ped
            num_peds_in_seq.append(num_peds_considered)
            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
            seq_list.append(curr_seq[:num_peds_considered])
            seq_list_rel.append(curr_seq_rel[:num_peds_considered])

num_seq = len(seq_list)
seq_list = np.concatenate(seq_list, axis=0)
seq_list_rel = np.concatenate(seq_list_rel, axis=0)
loss_mask_list = np.concatenate(loss_mask_list, axis=0)
non_linear_ped = np.asarray(non_linear_ped)
obs_traj = torch.from_numpy(seq_list[:, :, :obs_len]).type(torch.float)
pred_traj = torch.from_numpy(seq_list[:, :, obs_len:]).type(torch.float)
obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :obs_len]).type(torch.float)
loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
