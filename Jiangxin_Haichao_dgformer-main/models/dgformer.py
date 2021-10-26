import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import PySimpleGUI

from data.data_loarder import getdata
from data.data_loarder import getdata2
import torch.utils.data as Data
from models.utils import get_index_dgformer, links_laplace_matrces, links_to_sim_adjs, links_to_triplets
from models.utils import auc_link_prediction, precision_link_prediction
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# ==============================================================================================================


BATCH_SIZE = 20
EPOCH = 1000  # 模型训练次数

# ==============================================================================================================

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据构建
R = 500
S = 500
data = getdata2(train=True, us=False, r=R, s=S)
data_adjs = getdata2(train=True, us=True, r=R, s=S)
loader2 = Data.DataLoader(data_adjs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=data,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=False,  # 要不要打乱数据 (打乱比较好)
    drop_last=False
)
# ===========================================================================================================

d_ff = 256  # FeedForward 512->2048->512 做特征提取的
d_k = d_v = 16  # K(K=Q)和V的维度 Q和K的维度需要相同，这里为了方便让K=V
n_layers = 1  # Encoder and Decoder Layer Block的个数 6
n_heads = 1
d_model = data.nodes * 2

# 计算评价指标要采样的次数
# N = int(data.len * 0.01)
# L = int(data.nodes * 0.01)
N = L = R * BATCH_SIZE


# ====================================================================================================
# 以下是DGformer模型的各个模块

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 用来改变维度的，因为得到的数据是经过embedding的，需要原始维度
class DimChange(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(DimChange, self).__init__()
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x.view(-1))
        return x


dimc = DimChange(n_feature=data.nodes * 2, n_hidden=data.nodes, n_output=1).to(device)


def get_attn_pad_mask(seq_q, seq_k):
    new_seq_q = torch.zeros(len(seq_q), data.r).to(device)
    new_seq_k = torch.zeros(len(seq_q), data.r).to(device)
    # 下面这部分是神经网络，[8,5,56]->[8,5]
    for i in range(len(seq_q)):
        for j in range(data.r):
            new_seq_q[i][j] = dimc(seq_q[i][j])
            new_seq_k[i][j] = dimc(seq_k[i][j])
    new_seq_q, new_seq_k = new_seq_q.to(device), new_seq_k.to(device)
    batch_size, len_q = new_seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = new_seq_k.size()
    pad_attn_mask = new_seq_k.data.eq(0).unsqueeze(1)  # True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


# 标量点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.sim_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,
                                  bias=True).to(device)  # 卷积核大小和相似度矩阵一样大小

    def forward(self, Q, K, V, attn_mask, link_sim_mat):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ############################################################################################
        # 边的相似度矩阵嵌入到scores里面
        # 此处注释，即可做消融实验
        link_sim_mat = link_sim_mat.unsqueeze(1).float()  # unsqueeze(dim)在dim维度加一维
        sim_link = self.sim_conv(link_sim_mat)
        # sim_link = sim_link.squeeze(1)
        sim_link = sim_link.repeat(1, n_heads, 1, 1)

        # sim_link = sim_link.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = scores + sim_link
        ############################################################################################

        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度v做softmax
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context, attn


# 多头注意力模块
class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, link_sim_mat):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # ############################################################################################
        # # 边的相似度矩阵嵌入到attn_mask里面
        # link_sim_mat = link_sim_mat.unsqueeze(1)    # unsqueeze(dim)在dim维度加一维
        # sim_link = self.sim_conv(link_sim_mat)
        # sim_link = sim_link.reshape((BATCH_SIZE, data.r, data.r))
        # sim_link = sim_link.unsqueeze(1).repeat(1, n_heads, 1, 1)
        ##################################################

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, link_sim_mat)
        # 下面将不同头的输出向量拼接在一起
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


# 前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


# link_feature
class LinkFeature(nn.Module):

    def __init__(self, node_num):
        super().__init__()
        self.node_num = node_num
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=self.node_num * 2, out_features=100, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100, bias=True),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=self.node_num * 2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.hidden1(x)
        h2 = self.hidden2(h1)
        output1 = self.hidden3(h2)
        return output1


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask, link_sim_mat):
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, link_sim_mat)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# 解码层
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, link_sim_mat):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask, link_sim_mat)  # 这里的Q,K,V全是Decoder自己的输入
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask,
                                                      link_sim_mat)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


# 编码模块
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.link_embeding1 = LinkFeature(data.nodes)
        self.link_embeding2 = LinkFeature(data.nodes)
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, link_sim_mat, link_feature_input_v1, link_feature_input_v2):
        """
        enc_inputs: 1, 2 X data.nums
        link_feature_input_v1: 1, 2*data.nums
        """
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        # 由于我们的数据输入时就已经经过embedding，所以省略transformer里面的embedding
        #########################################################################################

        for batch in range(link_feature_input_v1.shape[0]):
            link_feature = self.link_embeding1(link_feature_input_v1[batch]) + self.link_embeding2(
                link_feature_input_v2[batch])
            enc_inputs[batch] = enc_inputs[batch] + link_feature
        #########################################################################################
        enc_outputs = enc_inputs
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask,
                                               link_sim_mat)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns, link_sim_mat


# 解码模块
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs, link_sim_mat):
        # dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = dec_inputs
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # Decoder的Block是上一个Block的输出dec_outputs(变化)和Encoder网络的输出enc_outputs(固定)
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask, link_sim_mat)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# =============================================================================================================================


# =============================================================================================================================
# 这是DGformer的模型
class DGformer(nn.Module):
    def __init__(self):
        super(DGformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, d_model, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs, link_sim_mat, link_feature_input_v1, link_feature_input_v2):
        enc_outputs, enc_self_attns, link_sim_mat = self.encoder(enc_inputs, link_sim_mat, link_feature_input_v1,
                                                                 link_feature_input_v2)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, link_sim_mat)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


# =============================================================================================================================

# ============================================model===============================================================
# 实例化模型 损失函数和优化器
model = DGformer().to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  # 用adam的话效果不好
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ====================================================================================================
def train():
    print('数据加载完成！')
    link_sim_mat_list = []
    link_feature_input_v1_list = []
    link_feature_input_v2_list = []
    for epoch in range(EPOCH):
        for (step, batch_adj), (_, batch_links) in zip(enumerate(loader2), enumerate(loader)):

            adjs_x = torch.squeeze(batch_adj[0], dim=1)
            samples = batch_links[0]
            labels = batch_links[1]

            begin = time.time()

            samples, labels = samples.to(device), labels.to(device)

            if epoch == 0:
                link_sim_mat = links_to_sim_adjs(samples, adjs=adjs_x)

                link_feature_input_v1, link_feature_input_v2 = links_to_triplets(samples)
                link_sim_mat_list.append(link_sim_mat)
                link_feature_input_v1_list.append(link_feature_input_v1)
                link_feature_input_v2_list.append(link_feature_input_v2)
            else:
                link_sim_mat = link_sim_mat_list[step]
                link_feature_input_v1 = link_feature_input_v1_list[step]
                link_feature_input_v2 = link_feature_input_v2_list[step]

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(samples, labels, link_sim_mat,
                                                                           link_feature_input_v1, link_feature_input_v2)

            outputs1 = outputs[:, :, :data.nodes]
            outputs2 = outputs[:, :, data.nodes:]
            labels1 = labels[:, :, :data.nodes]
            labels2 = labels[:, :, data.nodes:]

            Lps1 = links_laplace_matrces(outputs1, outputs2)
            Lps2 = links_laplace_matrces(labels1, labels2)
            L1 = Lps1 - Lps2
            L2 = torch.norm(L1, dim=(1, 2))
            loss1 = torch.sum(L2)

            # 为了使用交叉熵计算损失,reshape
            outputs1 = outputs1.reshape(outputs.shape[0] * data.r, data.nodes)
            outputs2 = outputs2.reshape(outputs.shape[0] * data.r, data.nodes)
            labels1 = labels1.reshape(outputs.shape[0] * data.r, data.nodes)
            labels2 = labels2.reshape(outputs.shape[0] * data.r, data.nodes)

            labels1 = get_index_dgformer(labels1)
            labels2 = get_index_dgformer(labels2)

            loss = criterion(outputs1, labels1) + criterion(outputs2, labels2) + loss1
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            print('Epoch:', '%01d' % (epoch), step, 'loss =', '{:.6f}'.format(loss))
            end = time.time()
            print(end - begin)

            # loss_list.append(loss.item())
            # plt.plot([i for i in range(len(loss_list))], loss_list)
            # plt.show()
            # plt.pause(0.1)
    # 保存模型
    torch.save(model, 'dgformer.pkl')  # 保存整个模型


def eval():
    data = getdata2(train=False, us=False, r=R, s=S)
    loader = Data.DataLoader(
        dataset=data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        # num_workers=1
    )
    data_adjs = getdata2(train=False, us=True, r=R, s=S)
    loader2 = Data.DataLoader(data_adjs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = torch.load('dgformer.pkl').to(device)
    auc_list, precision_list = [], []
    with torch.no_grad():
        for (step, batch_adj), (_, batch_links) in zip(enumerate(loader2), enumerate(loader)):
            adjs_x = torch.squeeze(batch_adj[0])
            samples = batch_links[0]
            labels = batch_links[1]

            link_sim_mat = links_to_sim_adjs(samples, adjs=adjs_x)
            link_feature_input_v1, link_feature_input_v2 = links_to_triplets(samples)

            link_sim_mat = link_sim_mat.to(device)
            link_feature_input_v1 = link_feature_input_v1.to(device)
            link_feature_input_v2 = link_feature_input_v2.to(device)
            samples = samples.to(device)
            labels = labels.to(device)

            output, _, _, _ = model(samples, labels, link_sim_mat, link_feature_input_v1, link_feature_input_v2)

            # output, _, _, _ = model(samples, labels)
            output, labels = output.cpu().data.numpy(), labels.cpu().data.numpy()
            auc = auc_link_prediction(labels, output, n=N)
            precision = precision_link_prediction(labels, output, l=L)
            auc_list.append(auc)
            precision_list.append(precision)
            print('step:',step,'AUC:',auc,'precision:',precision)
        auc_result = np.mean(np.array(auc_list))
        precision_result = np.mean(np.array(precision_list))
        print(auc_result, precision_result)


# ====================================================================================================
# 模型训练
if __name__ == '__main__':
    # loss_list = []
    # plt.figure(figsize=(5, 4))
    # plt.ion()
    train()
    eval()
    PySimpleGUI.popup(os.path.basename(sys.argv[0]))
