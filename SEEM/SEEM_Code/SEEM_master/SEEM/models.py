import torch
import torch.nn as nn
import numpy as np
from sgan.utils import relative_to_abs, get_dset_path
try:
    import ipdb
except:
    import pdb as ipdb

#TODO:无图像数据的处理

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            #对于2d或3d输入进行BN。在训练时，该层计算每次输入的均值和方差，并进行平行移动。
            #移动平均默认的动量为0.1。在验证时，训练求得的均值/方差将用于标准化验证数据
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

#TODO:计算随机微分方程
def get_sgld_proposal(z, noise_input, netG, netE, seq_start_end, obs_traj, obs_traj_rel, beta=1., alpha=.01):
    z.requires_grad_(True)
    #TODO:需要generatorSO的noise_input
    e_z_front = netG(torch.unsqueeze(torch.cat([noise_input, z], dim=1), 0), seq_start_end, obs_traj, obs_traj_rel)
    pred_traj_fake = relative_to_abs(e_z_front, obs_traj[-1])
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, e_z_front], dim=0)
    e_z = netE(traj_fake, traj_fake_rel, seq_start_end)
    del_e_z = torch.autograd.grad(
        outputs=e_z, inputs=z,
        grad_outputs=torch.ones_like(e_z),
        only_inputs=True
    )[0]

    eps = torch.randn_like(z) * np.sqrt(alpha * 2)
    z_prime = (z - alpha * del_e_z + eps).detach()
    #TODO:调试11
    #ipdb.set_trace()
    return e_z, del_e_z, z_prime
#TODO: 加上generatorSO返回的noise_input
def MALA_corrected_sampler(netG, netE, args, shape, noise_input, seq_start_end, obs_traj, obs_traj_rel, return_ratio=False):
    beta = args.temp if hasattr(args, 'temp') else 1.
    z = torch.randn(*shape).cuda()
    #ipdb.set_trace()
    for i in range(2):#args.mcmc_iters = 2, 調試一下看看增加iter能否提高精度
        e_z, del_e_z, z_prime = get_sgld_proposal(z, noise_input, netG, netE, seq_start_end, obs_traj, obs_traj_rel)
        e_z_prime, del_e_z_prime, _ = get_sgld_proposal(z_prime, noise_input, netG, netE, seq_start_end, obs_traj, obs_traj_rel)

        log_q_zprime_z = (z_prime - z + args.alpha * del_e_z).norm(2, dim=1)
        log_q_zprime_z *= -1. / (4 * args.alpha)

        log_q_z_zprime = (z - z_prime + args.alpha * del_e_z_prime).norm(2, dim=1)
        log_q_z_zprime *= -1. / (4 * args.alpha)

        log_ratio_1 = -e_z_prime + e_z  # log [p(z_prime) / p(z)]
        log_ratio_2 = log_q_z_zprime - log_q_zprime_z  # log [q(z | z_prime) / q(z_prime | z)]
        #print(log_ratio_1.mean().item(), log_ratio_2.mean().item())

        ratio = (log_ratio_1.squeeze(1) + log_ratio_2).exp().clamp(max=1)
        #TODO:
        ratio = ratio.mean()
        # print(ratio.mean().item())
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()
        #TODO:调试15
        #ipdb.set_trace()
        z = (z_prime * mask + z * (1 - mask)).detach()
    #TODO:调试12
    #ipdb.set_trace()

    if return_ratio:
        return z, ratio
    else:
        return z.detach()

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

#TODO: 互信息计算网络
class StatisticsNetwork(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, 1)
        )
    def forward(self, x, z):
        #ipdb.set_trace()
        x = torch.cat([x, z], -1)
        return self.main(x).squeeze(-1)

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                #池化的方法
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                #池化的方法
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            # mlp 层
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        #ipdb.set_trace()
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
            #stack则会增加新的维度。 
            #如对两个1*2维的tensor在第0个维度上stack，则会变为2*1*2的tensor；
            #在第1个维度上stack，则会变为1*2*2的tensor。
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        # 3*3 边界         
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2   #neighborhood_size=2.0
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2  
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        #                  ?????????????????????????
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *  #grid_size=8
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h

#TODO:init里应该加入netH的参数
#TODO:SO for step one
class TrajectoryGeneratorSO(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGeneratorSO, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        # TODO:self.pred_len = 12
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
    #TODO:从add_noise()改为get_noise_shape()
    def get_noise_shape(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim
        #TODO:把noise_shape返回给MALASampler
        return noise_shape
        # if user_noise is not None:
        #     z_decoder = user_noise
        # else:
        #     z_decoder = get_noise(noise_shape, self.noise_type)

        # if self.noise_mix_type == 'global':
        #     _list = []
        #     for idx, (start, end) in enumerate(seq_start_end):
        #         start = start.item()
        #         end = end.item()
        #         _vec = z_decoder[idx].view(1, -1)
        #         _to_cat = _vec.repeat(end - start, 1)
        #         _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        #     decoder_h = torch.cat(_list, dim=0)
        #     return decoder_h

        # decoder_h = torch.cat([_input, z_decoder], dim=1)
        # #TODO:增加了噪声返回
        # return decoder_h, z_decoder

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
        #TODO:调试5
        #
        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        #TODO:增加了decoder_z，用于返回增加的噪声
        #TODO:从这一下要放在generator外，generator_step里
        #TODO:generatorSO需要返回的是noise_input和get_noise_shape()的返回值noise_shape
        #TODO:add_noise()改成get_noise_shape()    
        noise_shape = self.get_noise_shape(noise_input, seq_start_end, user_noise=user_noise)
        #decoder_h = torch.unsqueeze(decoder_h, 0)
        #TODO:这里预计是generatorSO和generatorST的分界
        #ipdb.set_trace()
        return noise_input, noise_shape
        # decoder_c = torch.zeros(
        #     self.num_layers, batch, self.decoder_h_dim
        # ).cuda()

        # state_tuple = (decoder_h, decoder_c)
        # last_pos = obs_traj[-1]
        # last_pos_rel = obs_traj_rel[-1]
        # # Predict Trajectory

        # decoder_out = self.decoder(
        #     last_pos,
        #     last_pos_rel,
        #     state_tuple,
        #     seq_start_end,
        # )
        # pred_traj_fake_rel, final_decoder_h = decoder_out
        # #TODO: 返回值增加了decoder_z
        # return pred_traj_fake_rel, decoder_z

#TODO:ST for step two
class TrajectoryGeneratorST(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGeneratorST, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        # TODO:self.pred_len = 12
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        #TODO:不需要encoder
        # self.encoder = Encoder(
        #     embedding_dim=embedding_dim,
        #     h_dim=encoder_h_dim,
        #     mlp_dim=mlp_dim,
        #     num_layers=num_layers,
        #     dropout=dropout
        # )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )
        #TODO:不需要pooling层
        # if pooling_type == 'pool_net':
        #     self.pool_net = PoolHiddenNet(
        #         embedding_dim=self.embedding_dim,
        #         h_dim=encoder_h_dim,
        #         mlp_dim=mlp_dim,
        #         bottleneck_dim=bottleneck_dim,
        #         activation=activation,
        #         batch_norm=batch_norm
        #     )
        # elif pooling_type == 'spool':
        #     self.pool_net = SocialPooling(
        #         h_dim=encoder_h_dim,
        #         activation=activation,
        #         batch_norm=batch_norm,
        #         dropout=dropout,
        #         neighborhood_size=neighborhood_size,
        #         grid_size=grid_size
        #     )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        # if self.mlp_decoder_needed():
        #     mlp_decoder_context_dims = [
        #         input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
        #     ]

        #     self.mlp_decoder_context = make_mlp(
        #         mlp_decoder_context_dims,
        #         activation=activation,
        #         batch_norm=batch_norm,
        #         dropout=dropout
            # )
    #TODO:不需要加噪声
    # def add_noise(self, _input, seq_start_end, user_noise=None):
    #     """
    #     Inputs:
    #     - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
    #     - seq_start_end: A list of tuples which delimit sequences within batch.
    #     - user_noise: Generally used for inference when you want to see
    #     relation between different types of noise and outputs.
    #     Outputs:
    #     - decoder_h: Tensor of shape (_, decoder_h_dim)
    #     """
    #     if not self.noise_dim:
    #         return _input

    #     if self.noise_mix_type == 'global':
    #         noise_shape = (seq_start_end.size(0), ) + self.noise_dim
    #     else:
    #         noise_shape = (_input.size(0), ) + self.noise_dim

    #     if user_noise is not None:
    #         z_decoder = user_noise
    #     else:
    #         z_decoder = get_noise(noise_shape, self.noise_type)

    #     if self.noise_mix_type == 'global':
    #         _list = []
    #         for idx, (start, end) in enumerate(seq_start_end):
    #             start = start.item()
    #             end = end.item()
    #             _vec = z_decoder[idx].view(1, -1)
    #             _to_cat = _vec.repeat(end - start, 1)
    #             _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
    #         decoder_h = torch.cat(_list, dim=0)
    #         return decoder_h

    #     decoder_h = torch.cat([_input, z_decoder], dim=1)
    #     #TODO:增加了噪声返回
    #     return decoder_h, z_decoder
    #TODO:不需要mlp
    # def mlp_decoder_needed(self):
    #     if (
    #         self.noise_dim or self.pooling_type or
    #         self.encoder_h_dim != self.decoder_h_dim
    #     ):
    #         return True
    #     else:
    #         return False

    def forward(self, decoder_h, seq_start_end, obs_traj, obs_traj_rel, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        # batch = obs_traj_rel.size(1)
        # Encode seq
        #final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        # if self.pooling_type:
        #     end_pos = obs_traj[-1, :, :]
        #     pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
        #     # Construct input hidden states for decoder
        #     mlp_decoder_context_input = torch.cat(
        #         [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        # else:
        #     mlp_decoder_context_input = final_encoder_h.view(
        #         -1, self.encoder_h_dim)
        #TODO:调试5
        #ipdb.set_trace()
        # Add Noise
        # if self.mlp_decoder_needed():
        #     noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        # else:
        #     noise_input = mlp_decoder_context_input
        #TODO:增加了decoder_z，用于返回增加的噪声    
        # decoder_h, decoder_z = self.add_noise(
        #     noise_input, seq_start_end, user_noise=user_noise)
        # decoder_h = torch.unsqueeze(decoder_h, 0)
        batch = obs_traj_rel.size(1)
        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out
        #TODO: 返回值不需要decoder_z了
        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
 
