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
    # 1:1:1:1:1:1:1
    while batch_size % 7 != 0:
        batch_size += 1

    # if batch_size % 7 == 0:
    idx1 = batch_size // 10 * 1
    idx2 = batch_size // 10 * 1
    idx3 = batch_size // 10 * 1
    idx4 = batch_size // 10 * 1
    idx5 = batch_size // 10 * 4
    idx6 = batch_size // 10 * 1
    idx7 = batch_size // 10 * 1

    for i in range(idx1):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        y2 = np.linspace(8, 22, seq_length // 2)
        y2 = y2 + np.random.randn(seq_length // 2) * 0.1
        x2 = np.zeros(y2.shape) + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]
        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx2):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + math.tan(math.pi/6) * x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = -x2 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx3):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)
        x2 = x2 + np.random.randn(seq_length // 2) * 0.1
        y2 = 8 + math.tan(math.pi/6) * x2 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx4):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)

        y2 = 8 + np.random.randn(seq_length // 2) * 0.1
        x2 = -x2 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx5):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)
        x2 = x2 + np.random.randn(seq_length // 2) * 0.1
        y2 = 8 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx6):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)
        x2 = x2 + np.random.randn(seq_length // 2) * 0.1
        y2 = 8 + math.tan(math.pi/3) * x2 + np.random.randn(seq_length // 2) * 0.1

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    curr_seq = []
    for i in range(idx7):
        y1 = np.linspace(1, 8, seq_length // 2)
        y1 = y1 + np.random.randn(seq_length // 2) * 0.1
        x1 = np.zeros(y1.shape) + np.random.randn(seq_length // 2) * 0.1

        x2 = np.linspace(0, 8, seq_length // 2)
        y2 = 8 + math.tan(math.pi / 3) * x2 + np.random.randn(seq_length // 2) * 0.1
        x2 = -x2 + np.random.randn(seq_length // 2) * 0.1


        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        x_rel[1:] = x[1:] - x[:-1]
        y_rel[1:] = y[1:] - y[:-1]

        tmp = np.array([x, y])
        curr_seq.append(tmp)
        curr_seq_rel.append(np.array([x_rel, y_rel]))
        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, 8, threshold))
    curr_seq_all_list.append(np.array(curr_seq))
    # curr_seq = []
    # for i in range(1,batch_size+1):
    #     if i % 10 == 0 or i % 10 == 1 :
    #         y1 = np.linspace(1 ,8, seq_length//2)
    #         y1 = y1+np.random.randn(seq_length//2)*0.1
    #         x1 = np.zeros(y1.shape)+np.random.randn(seq_length//2)*0.1
    #
    #         y2 = np.linspace(8 ,16, seq_length//2)
    #         y2 = y2+np.random.randn(seq_length//2)*0.1
    #         x2 = np.zeros(y2.shape)+np.random.randn(seq_length//2)*0.1
    #
    #     elif i % 10 ==2 or i % 10 ==3: #label 1
    #         y1 = np.linspace(1 ,8, seq_length//2)
    #         y1 = y1+np.random.randn(seq_length//2)*0.1
    #         x1 = np.zeros(y1.shape)+np.random.randn(seq_length//2)*0.1
    #
    #         x2 = np.linspace(0 ,8, seq_length//2)
    #
    #         y2 = 8+x2+np.random.randn(seq_length//2)*0.1
    #         x2 = -x2+np.random.randn(seq_length//2)*0.1
    #
    #     else:#label 2
    #         y1 = np.linspace(1 ,8, seq_length//2)
    #         y1 = y1+np.random.randn(seq_length//2)*0.1
    #         x1 = np.zeros(y1.shape)+np.random.randn(seq_length//2)*0.1
    #
    #         x2 = np.linspace(0 ,8, seq_length//2)
    #
    #         x2 = x2+np.random.randn(seq_length//2)*0.1
    #         y2 = 8+x2+np.random.randn(seq_length//2)*0.1
    #
    #
    #     x=np.concatenate((x1,x2),axis=0)
    #     y=np.concatenate((y1,y2),axis=0)
    #
    #     x_rel[1:]=x[1:]-x[:-1]
    #     y_rel[1:]=y[1:]-y[:-1]
    #     tmp= np.array([x, y])
    #     curr_seq.append( tmp)
    #     curr_seq_rel.append(np.array([x_rel,y_rel]))
    #     curr_loss_mask.append(np.ones(seq_length))
    #
    #     _non_linear_ped.append(poly_fit(tmp, 8, threshold))

    curr_seq = np.array(curr_seq_all_list)
    curr_seq_rel = np.array(curr_seq_rel)
    curr_loss_mask = np.array(curr_loss_mask)
    _non_linear_ped = np.array(_non_linear_ped)

    #    sample = np.array(sample).transpose((2, 0, 1))
    #    obs_traj_rel = np.zeros(sample[:,:2,:].shape)

    #    ipdb.set_trace()
    #    obs_traj_rel[:,:, 1:] = sample[:,:2,1:]-sample[:,:2,:-1]
    return curr_seq, curr_seq_rel, curr_loss_mask, _non_linear_ped


# curr_seq,curr_seq_rel,curr_loss_mask,_non_linear_ped=generate_x_y_data_label(batch_size=1)

def plot():
    curr_seq, curr_seq_rel, curr_loss_mask, _non_linear_ped = generate_x_y_data_label(batch_size=10)
    if curr_seq is None:
        return None
    for idx in range(len(curr_seq)):
        for i in range(len(curr_seq[idx])):
            if idx == 0:
                if i == 0:
                    plt.scatter(curr_seq[idx][i][0][:9], curr_seq[idx][i][1][:9], c='r', marker='o')
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#6495ED', marker='o')
            elif idx == 1:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#00FFFF', marker='o')
            elif idx == 2:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#008000', marker='o')
            elif idx == 3:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#FFA500', marker='o')
            elif idx == 4:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#FF8C00', marker='o')
            elif idx == 5:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#0000FF', marker='o')
            elif idx == 6:
                plt.scatter(curr_seq[idx][i][0][9:], curr_seq[idx][i][1][9:], c='#A52A2A', marker='o')
    plt.savefig('wuchashu.png')
    # plt.show()

plot()