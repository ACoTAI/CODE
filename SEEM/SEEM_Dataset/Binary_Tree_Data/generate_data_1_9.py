import numpy as np
import requests
import random
import math
import torch
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
    #label 0
    threshold=0.002
    x_rel = np.zeros(seq_length)
    y_rel= np.zeros(seq_length)


    curr_seq=[]
    curr_seq_rel=[]
    curr_loss_mask=[]
    _non_linear_ped=[]

    for i in range(1,batch_size+1):        
        y1 = np.linspace(1 ,8, seq_length//2)
        y1 = y1+np.random.randn(seq_length//2)*0.1
        x1 = np.zeros(y1.shape)+np.random.randn(seq_length//2)*0.1

        if i % 10 == 0: #zuo:you=1:9
            x2 = np.linspace(1 ,8, seq_length//2)
            x2 = -x2+np.random.randn(seq_length//2)*0.1
            y2 = 8+np.random.randn(seq_length//2)*0.1
        else:
            x2 = np.linspace(1 ,8, seq_length//2)
            x2 = x2+np.random.randn(seq_length//2)*0.1
            y2 = 8+np.random.randn(seq_length//2)*0.1

        x=np.concatenate((x1,x2),axis=0)
        y=np.concatenate((y1,y2),axis=0)  

        x_rel[1:]=x[1:]-x[:-1]
        y_rel[1:]=y[1:]-y[:-1]

        tmp= np.array([x, y])
        curr_seq.append( tmp)


        
        curr_seq_rel.append(np.array([x_rel,y_rel]))


        curr_loss_mask.append(np.ones(seq_length))

        _non_linear_ped.append(poly_fit(tmp, seq_length, threshold))

    curr_seq=np.array(curr_seq)
    curr_seq_rel=np.array(curr_seq_rel)
    curr_loss_mask=np.array(curr_loss_mask)
    _non_linear_ped=np.array(_non_linear_ped)

#    sample = np.array(sample).transpose((2, 0, 1))
#    obs_traj_rel = np.zeros(sample[:,:2,:].shape)
    
#    ipdb.set_trace()
#    obs_traj_rel[:,:, 1:] = sample[:,:2,1:]-sample[:,:2,:-1]
    return curr_seq,curr_seq_rel,curr_loss_mask,_non_linear_ped

#curr_seq,curr_seq_rel,curr_loss_mask,_non_linear_ped=generate_x_y_data_label(batch_size=1)




    
    
