import argparse
import gc#垃圾回收
import logging
import os
import sys
import time
import ipdb


from collections import defaultdict
#这个模块实现了特定目标的容器，
#以提供Python标准内建容器 dict , list , set , 和 tuple 的替代选择。
#defaultdirt是字典的子类，提供了一个工厂函数，为字典查询提供一个默认值
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGeneratorSO, TrajectoryGeneratorST, TrajectoryDiscriminator
from sgan.models import StatisticsNetwork, MALA_corrected_sampler, get_sgld_proposal
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='hotel', type=str)
parser.add_argument('--delim', default=' ')  # 在数据集的文件中使用的分隔符,默认为' '
parser.add_argument('--loader_num_workers', default=8, type=int)
parser.add_argument('--obs_len', default=8, type=int) # 输入轨迹中的时间步数.默认值为8.
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)  #制作数据集时要跳过的帧数.
                                                    #对于例如 如果数据集中的Sequence1来自Frame1 - FrameN且skip = 2
                                                    #那么Sequence2将来自Frame3 - FrameN + 2.默认值为1

# Optimization
parser.add_argument('--batch_size', default=128, type=int)       #迭代次数
parser.add_argument('--num_iterations', default=10000, type=int)#迭代次数
parser.add_argument('--num_epochs', default=400, type=int)      #遍历训练集的次数 origial1000/800

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)    #映射维度
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)  #BN批规范化
parser.add_argument('--mlp_dim', default=1024, type=int)        #Multilayer perceptron  多层感知器维度
# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)           # An iteration consists of g_steps forward backward pass on the
parser.add_argument('--alpha', type=float, default=.01)                                                              # generator. Default is 1.
# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int) # Output dimensions of the pooled vector for Pool Net

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)
parser.add_argument('--best_k', default=2, type=int)            #TODO:sophie中用的2
parser.add_argument('--h_learning_rate', default=5e-4, type=float)#TODO:互信息网络

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)       #TODO:sophie用的20
parser.add_argument('--checkpoint_every', default=200, type=int) #TODO:原来为100，修改为10
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument('--lamdba_l2', default=0.2, type=float)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="1", type=str)
#parser.add_argument('--args.mcmc_iters', default=2, type=int)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    #TODO:sophie这里加了image相关路径
    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    #TODO:调试1

    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )
    #TODO:generator step one
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

    generatorSO.apply(init_weights)
    generatorSO.type(float_dtype).train()
    logger.info('Here is the generatorSO:')
    logger.info(generatorSO)
    #TODO:generator step two
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
    
    generatorST.apply(init_weights)
    generatorST.type(float_dtype).train()
    logger.info('Here is the generatorST:')
    logger.info(generatorST)
    
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

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)
    #TODO:调试7
    #ipdb.set_trace()
    netH = StatisticsNetwork(z_dim = 2*args.noise_dim[0] + 4*args.pred_len, dim=512)#TODO:修改维度
    netH.apply(init_weights)
    netH.type(float_dtype).train()
    logger.info('Here is the netH:')
    logger.info(netH)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss
    #TODO:增加了optimizer_gso和optimizer_gst
    optimizer_gso = optim.Adam(generatorSO.parameters(), lr=args.g_learning_rate)
    optimizer_gst = optim.Adam(generatorST.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
    optimizer_h = optim.Adam(netH.parameters(), lr=args.h_learning_rate)#TODO:互信息计算网络的优化器


    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generatorSO.load_state_dict(checkpoint['gso_state'])
        generatorST.load_state_dict(checkpoint['gst_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        #TODO:gso&gst
        optimizer_gso.load_state_dict(checkpoint['gso_optim_state'])
        optimizer_gst.load_state_dict(checkpoint['gst_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            #TODO:增加norm_gso和norm_gst
            'norm_gso': [],
            'norm_gst': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            #TODO:gso&gst
            'gso_state': None,
            'gst_state': None,
            'gso_optim_state': None,
            'gst_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'gso_best_state': None,
            'gst_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'gso_best_nl_state': None,
            'gst_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        #TODO:调试2
        #ipdb.set_trace()
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generatorSO, generatorST,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generatorSO, generatorST,
                                          discriminator, netH, g_loss_fn,
                                          optimizer_gso, optimizer_gst, optimizer_h)
                                          #TODO:增加了netH,optimizer_h,optimizer_gso,optimizer_gst
                #TODO:gso和gst
                checkpoint['norm_gso'].append(
                    get_total_norm(generatorSO.parameters())
                )
                checkpoint['norm_gst'].append(
                    get_total_norm(generatorST.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                #TODO:调试9
                #ipdb.set_trace()
                #TODO:gso&gst
                metrics_val = check_accuracy(
                    args, val_loader, generatorSO, generatorST, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                #TODO:gso&gst
                metrics_train = check_accuracy(
                    args, train_loader, generatorSO, generatorST, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    #TODO:gso&gst
                    checkpoint['gso_best_state'] = generatorSO.state_dict()
                    checkpoint['gst_best_state'] = generatorST.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    #TODO:gso&gst
                    checkpoint['gso_best_nl_state'] = generatorSO.state_dict()
                    checkpoint['gst_best_nl_state'] = generatorST.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                #TODO:gso&gst
                checkpoint['gso_state'] = generatorSO.state_dict()
                checkpoint['gst_state'] = generatorST.state_dict()
                #TODO:gso&gst
                checkpoint['gso_optim_state'] = optimizer_gso.state_dict()
                checkpoint['gst_optim_state'] = optimizer_gst.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                #TODO:gso&gst
                key_blacklist = [
                    'gso_state', 'gst_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'gso_optim_state', 'gst_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


def discriminator_step(
    args, batch, generatorSO, generatorST, discriminator, d_loss_fn, optimizer_d
):
    #TODO:调试3
    #ipdb.set_trace()
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    #TODO:generator的返回值增加了z_noise
    #将注释下一句换为注释下五句

    #generator_out, z_noise = generator(obs_traj, obs_traj_rel, seq_start_end)
    noise_input, noise_shape = generatorSO(obs_traj, obs_traj_rel, seq_start_end) #noise _input[876,120], z_noise[876,8]
    #ipdb.set_trace()
    z_noise = MALA_corrected_sampler(generatorST, discriminator, args, noise_shape, noise_input, seq_start_end, obs_traj, obs_traj_rel)
    #ipdb.set_trace()
    decoder_h = torch.cat([noise_input, z_noise], dim=1)
    decoder_h = torch.unsqueeze(decoder_h, 0)

    generator_out = generatorST(decoder_h, seq_start_end, obs_traj, obs_traj_rel)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
#TODO:增加了可视化
#-------------------------visualzation----------------------------------------------

    '''window_num = 4
    for i in range(window_num):
        previous_tra=obs_traj[:,i,:]
        previous_tra_x=previous_tra[:,0]
        previous_tra_y=previous_tra[:,1]
        previous_tra_x=previous_tra_x.cpu()
        previous_tra_y=previous_tra_y.cpu()
        previous_tra_x=previous_tra_x.numpy()
        previous_tra_y=previous_tra_y.numpy()
        groundtruth_tra=pred_traj_gt[:,i,:]
        groundtruth_tra_x=groundtruth_tra[:,0]
        groundtruth_tra_y=groundtruth_tra[:,1]
        groundtruth_tra_x=groundtruth_tra_x.cpu()
        groundtruth_tra_y=groundtruth_tra_y.cpu()
        groundtruth_tra_x=groundtruth_tra_x.numpy()
        groundtruth_tra_y=groundtruth_tra_y.numpy()

        plt.plot(previous_tra_x,previous_tra_y , "o--b")
        plt.plot(groundtruth_tra_x,groundtruth_tra_y , "o--r")
        predicted_tra=pred_traj_fake[:,i,:]
        predicted_tra_x=predicted_tra[:,0]
        predicted_tra_y=predicted_tra[:,1]
        predicted_tra_x=predicted_tra_x.cpu()
        predicted_tra_y=predicted_tra_y.cpu()

        predicted_tra_x=predicted_tra_x.detach().numpy()
        predicted_tra_y=predicted_tra_y.detach().numpy()
        plt.plot(predicted_tra_x,predicted_tra_y , "o--y")
        plt.axis([-17,17,-17,17])
        plt.title("Prediction trajectory")
        plt.savefig('./plot_d/'+str(i+d_dex*window_num)+'.jpg',dpi=400)

        plt.close()'''


#-----------------------------------------------------------------------------------
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += (data_loss * args.lamdba_l2)
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses

#TODO:增加netH和optimizer_h
#TODO:增加了generatorSO和generatorST
#TODO:增加了optimizer_gso和optimizer_gst
def generator_step(
    args, batch, generatorSO, generatorST, discriminator, netH, g_loss_fn, optimizer_gso, optimizer_gst, optimizer_h
):
    #TODO:调试4
    #ipdb.set_trace()
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []
    #TODO: 增加互信息loss积累list
    g_mi_loss_rel = []
    loss_mask = loss_mask[:, args.obs_len:]
    #TODO:label用于最小化二值交叉熵的计算
    #ipdb.set_trace()
    label = torch.zeros((seq_start_end[-1][-1]-seq_start_end[0][0]).item()).cuda()#TODO:依据batchsize而定
    label[: int(len(label)/2)].data.fill_(1)#TODO:是batchsize的一半

    for _ in range(args.best_k):
        #TODO: generator的返回值增加z_noise,以便在generator_step中计算互信息
        #将注释下一句换为注释下五句
        #generator_out, z_noise = generator(obs_traj, obs_traj_rel, seq_start_end)
        noise_input, noise_shape = generatorSO(obs_traj, obs_traj_rel, seq_start_end)
        #TODO:调试13
        #ipdb.set_trace()
        z_noise = MALA_corrected_sampler(generatorST, discriminator, args, noise_shape, noise_input, seq_start_end, obs_traj, obs_traj_rel)
        #TODO:调试14
        #ipdb.set_trace()
        decoder_h = torch.cat([noise_input, z_noise], dim=1)
        decoder_h = torch.unsqueeze(decoder_h, 0)
        generator_out = generatorST(decoder_h, seq_start_end, obs_traj, obs_traj_rel)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))
        #TODO: 每个batch的互信息计算
        #TODO:调试6
        #ipdb.set_trace()
        z_noise_bar = z_noise[torch.randperm(len(z_noise))]#TODO:z_noise第一维就是batchsize
        concat_x_pred = torch.cat([pred_traj_fake, pred_traj_fake], 0)
        concat_z_noise = torch.cat([z_noise, z_noise_bar], -1)
        mi_estimate = nn.BCEWithLogitsLoss()(netH(concat_x_pred.permute(1,0,2).reshape(len(concat_z_noise),-1).squeeze(), concat_z_noise).squeeze(), label)
        g_mi_loss_rel.append(mi_estimate)

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += (g_l2_loss_sum_rel * (1 - args.lamdba_l2))
    #TODO:多轮训练完成后，计算总体互信息loss

    #ipdb.set_trace()
    g_mi_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)

    #TODO:调试8
    #ipdb.set_trace()
    #TODO:dim按第一维
    g_mi_loss_rel = torch.stack(g_mi_loss_rel, dim=0)
    g_mi_loss_sum_rel = torch.sum(g_mi_loss_rel)
    losses['G_mi_loss_rel'] = g_mi_loss_sum_rel.item()
    loss += (g_mi_loss_sum_rel * args.lamdba_l2)

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += (discriminator_loss * args.lamdba_l2)
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()
    #TODO:optimizer_gso和optimizer_gst
    optimizer_gso.zero_grad()
    optimizer_gst.zero_grad()
    #TODO:增加netH优化器
    optimizer_h.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generatorSO.parameters(), generatorST.parameters(), args.clipping_threshold_g
        )
    #TODO:generator的两个部分要更新
    optimizer_gso.step()
    optimizer_gst.step()
    #TODO:对于计算互信息的网络H也同样要更新
    optimizer_h.step()

    return losses


def check_accuracy(
    args, loader, generatorSO, generatorST, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generatorSO.eval()
    generatorST.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]
            #TODO:增加了z_noise
            #将注释下一句换为注释下五句
            #pred_traj_fake_rel, z_noise = generator(obs_traj, obs_traj_rel, seq_start_end)
            noise_input, noise_shape = generatorSO(obs_traj, obs_traj_rel, seq_start_end)
            #TODO:调试16
            #ipdb.set_trace()
            with torch.enable_grad():
                generatorST.train()
                z_noise = MALA_corrected_sampler(generatorST, discriminator, args, noise_shape, noise_input, seq_start_end, obs_traj, obs_traj_rel)
                generatorST.eval()
            decoder_h = torch.cat([noise_input, z_noise], dim=1)
            decoder_h = torch.unsqueeze(decoder_h, 0)
            generator_out = generatorST(decoder_h, seq_start_end, obs_traj, obs_traj_rel)
            #TODO:调试10
            #ipdb.set_trace()
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generatorSO.train()
    generatorST.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
