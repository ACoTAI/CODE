import numpy as np
import requests
import random
import math
import torch
import matplotlib.pyplot as plt

try:
    import ipdb
except:
    import pdb as ipdb


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


def generate_x_y_data_label(batch_size):
    seq_length = 16

    threshold = 0.002
    x_rel = np.zeros(seq_length)
    y_rel = np.zeros(seq_length)
    curr_seq_all_list = []
    curr_seq = []
    curr_seq_rel = []
    curr_loss_mask = []
    _non_linear_ped = []
    # left : mid : right = 2 : 2 : 6
    # label 0
    # ipdb.set_trace()
    # 1:1
    if batch_size % 10 == 0:
        idx1 = batch_size // 10 * 1
        idx2 = batch_size // 10 * 9

    else:
        return None
    #1:1
    idx1_1 = idx1 // 10 * 1
    idx1_2 = idx1 // 10 * 9
    idx2_1 = idx2 // 10 * 1
    idx2_2 = idx2 // 10 * 9

    for _ in range(idx1_1):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = -x2 + np.random.randn(seq_length // 2) * 0.1
        x3 = np.linspace(8, 16, seq_length // 2)

        y3 = 16 + math.tan(math.pi / 8) * (x3 - 8) + np.random.randn(seq_length // 2) * 0.1
        x3 = -x3 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((y1, y2, y3), axis=0)
        tmp = np.array([x, y])
        curr_seq.append(tmp)
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx1_2):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = -x2 + np.random.randn(seq_length // 2) * 0.1
        x3 = np.linspace(8, 12, seq_length // 2)

        y3 = 16 + math.tan(math.pi / 8 * 3) * (x3 - 8) + np.random.randn(seq_length // 2) * 0.1
        x3 = -x3 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((y1, y2, y3), axis=0)

        tmp = np.array([x, y])
        curr_seq.append(tmp)
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for _ in range(idx2_2):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = x2 + np.random.randn(seq_length // 2) * 0.1
        x3 = np.linspace(8, 16, seq_length // 2)

        y3 = 16 + math.tan(math.pi / 8) * (x3 - 8) + np.random.randn(seq_length // 2) * 0.1
        x3 = x3 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((y1, y2, y3), axis=0)
        tmp = np.array([x, y])
        curr_seq.append(tmp)
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for _ in range(idx2_1):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = x2 + np.random.randn(seq_length // 2) * 0.1
        x3 = np.linspace(8, 12, seq_length // 2)

        y3 = 16 + math.tan(math.pi / 8 * 3) * (x3 - 8) + np.random.randn(seq_length // 2) * 0.1
        x3 = x3 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((y1, y2, y3), axis=0)
        tmp = np.array([x, y])
        curr_seq.append(tmp)
    curr_seq_all_list.append(np.array(curr_seq))

    curr_seq = np.array(curr_seq_all_list)

    return curr_seq#, curr_seq_rel, curr_loss_mask, _non_linear_ped


# curr_seq,curr_seq_rel,curr_loss_mask,_non_linear_ped=generate_x_y_data_label(batch_size=1)

def plot():
    curr_seq = generate_x_y_data_label(batch_size=100)
    if curr_seq is None:
        return None
    for idx in range(len(curr_seq)):
        for i in range(len(curr_seq[idx])):
            if idx == 0:
                if i == 0:
                    plt.scatter(curr_seq[idx][i][0][:9], curr_seq[idx][i][1][:9], c='r', marker='o')
                plt.scatter(curr_seq[idx][i][0][9:16], curr_seq[idx][i][1][9:16], c='#808080', marker='o')
                plt.scatter(curr_seq[idx][i][0][16:], curr_seq[idx][i][1][16:], c='g', marker='o')
            elif idx == 1:
                plt.scatter(curr_seq[idx][i][0][9:16], curr_seq[idx][i][1][9:16], c='#808080', marker='o')
                plt.scatter(curr_seq[idx][i][0][16:], curr_seq[idx][i][1][16:], c='b', marker='o')
            elif idx == 2:
                plt.scatter(curr_seq[idx][i][0][9:16], curr_seq[idx][i][1][9:16], c='y', marker='o')
                plt.scatter(curr_seq[idx][i][0][16:], curr_seq[idx][i][1][16:], c='#00CED1', marker='o')
            elif idx == 3:
                plt.scatter(curr_seq[idx][i][0][9:16], curr_seq[idx][i][1][9:16], c='y', marker='o')
                plt.scatter(curr_seq[idx][i][0][16:], curr_seq[idx][i][1][16:], c='#FFA500', marker='o')
            # elif idx == 4:
            #     plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#00CED1', marker='o')
    plt.savefig('wuchashu.png')
    # plt.show()

plot()