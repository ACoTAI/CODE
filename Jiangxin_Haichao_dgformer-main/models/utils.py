import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 取出位置为1的索引，作为训练target
def get_index_dgformer(labels):
    new_labels = torch.zeros(labels.shape[0], dtype=torch.int64).to(device)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] == 1:
                new_labels[i] = j
    return new_labels.to(device)


# ====================================================================================================
# 取出位置为1的索引，作为训练target
def get_index_transformer(labels, data):
    new_labels = labels.reshape(labels.shape[0], data.r * 2, data.nodes).cpu().numpy()
    index = []
    for i in range(len(new_labels)):
        for j in range(len(new_labels[0])):
            for k in range(len(new_labels[0][0])):
                if new_labels[i][j][k] == 1:
                    index.append(k)
    index = torch.LongTensor(index).reshape(len(labels), data.r * 2).to(device)
    return index


# ====================================================================================================


# 计算模型输出概率矩阵与真实邻接矩阵的AUC
def auc_link_prediction(A, A_predict, n):
    n1 = 0
    n2 = 0
    # print(np.nonzero(A))
    x, y, z = np.nonzero(A)  # 输出邻接矩阵中非0元素位置
    # print(x, y)
    x2 = np.argwhere(A == 0)  # 多少个为0的位置
    # print(x2)
    for i in range(n):
        index1 = np.random.randint(0, len(x), 1)  # 随机选择一个存在的边
        # print(1, index1)
        index2 = np.random.randint(0, x2.shape[0], 1)  # 随机选择一个不存在的边
        # print(2, index2)
        temp1 = A_predict[x[index1], y[index1], z[index1]]  # A_predict中存在的边数值
        temp2 = A_predict[x2[index2, 0], x2[index2, 1], x2[index2, 2]]  # A_predict中不存在的边数值
        if temp1 > temp2:
            n1 += 1
        if temp1 == temp2:
            n2 += 1

    return (n1 + 0.5 * n2) / n


# 计算模型输出概率矩阵与真实邻接矩阵的precision
def precision_link_prediction(A, a_predict, l):
    n = 0
    a_predict = np.reshape(a_predict, (-1))
    A = np.reshape(A, (-1))
    # print(a_predict)
    # print(1, A)

    # .argsort()是从小到大
    # a_predict_argsort是从大到小排列
    a_predict_argsort = (-a_predict).argsort()
    # print(a_predict_argsort)
    for i in range(l):
        # print(A[a_predict_argsort[i]])
        if A[a_predict_argsort[i]] == 1:
            n += 1
    # print(n)
    # print(L)
    precision = n / l
    return precision


def links_to_sim(links):
    l1 = links.shape[1]  # link number
    node_num = int(links.shape[2] / 2)
    link_sim_mats = torch.zeros([links.shape[0], l1, l1]).cuda()

    for s in range(links.shape[0]):
        links_of_samlpe = links[s, :, :]
        N1 = torch.where(links_of_samlpe[:, :node_num])[1] + 1  # 出发节点
        N2 = torch.where(links_of_samlpe[:, node_num:])[1] + 1  # 到达节点
        link_sim_mat = torch.zeros([l1, l1]).cuda()

        for i in range(l1):
            print('link sim calculate step:', s, 'total steps:', links.shape[0], 'link:', i)
            v1 = N1[i]  # link1出发节点
            v2 = N2[i]  # link1达到节点
            n12 = torch.where(N1 == v1, N2, torch.zeros_like(N1))
            n12 = n12[torch.where(n12)]  # v1节点的所有到达节点
            n11 = torch.where(N2 == v1, N1, torch.zeros_like(N1))
            n11 = n11[torch.where(n11)]  # v1节点的所有出发节点
            n1 = torch.cat((n11, n12)).unique()
            n21 = torch.where(N1 == v2, N2, torch.zeros_like(N1))
            n21 = n21[torch.where(n21)]  # v2节点的所有到达节点
            n22 = torch.where(N2 == v2, N1, torch.zeros_like(N1))
            n22 = n22[torch.where(n22)]  # v2节点的所有出发节点
            n2 = torch.cat((n21, n22)).unique()

            for j in range(l1):
                if j > i:
                    v3 = N1[j]  # link2出发节点
                    v4 = N2[j]  # link2达到节点
                    n32 = torch.where(N1 == v3, N2, torch.zeros_like(N1))
                    n32 = n32[torch.where(n32)]  # v3节点的所有到达节点
                    n31 = torch.where(N2 == v3, N1, torch.zeros_like(N1))
                    n31 = n31[torch.where(n31)]  # v3节点的所有出发节点
                    n3 = torch.cat((n31, n32)).unique()
                    n41 = torch.where(N1 == v4, N2, torch.zeros_like(N1))
                    n41 = n41[torch.where(n41)]  # v4节点的所有到达节点
                    n42 = torch.where(N2 == v4, N1, torch.zeros_like(N1))
                    n42 = n42[torch.where(n42)]  # v4节点的所有出发节点
                    n4 = torch.cat((n41, n42)).unique()
                    # print(n1, '\n', n2, '\n', n3, '\n', n4)

                    s1 = 0
                    s2 = 0
                    for v5 in n1:
                        if torch.where(n3 == v5)[0].shape[0] > 0:
                            s1 += 1

                    for v6 in n2:
                        if torch.where(n4 == v6)[0].shape[0] > 0:
                            s2 += 1

                    sim = s1 / (len(n1) + len(n3)) + s2 / (len(n2) + len(n4))

                    link_sim_mat[i, j] = sim
                    link_sim_mat[j, i] = sim

        link_sim_mats[s, :, :] = link_sim_mat
    return link_sim_mats


def links_to_sim_adjs(links, adjs):
    l1 = links.shape[1]  # link number
    node_num = int(links.shape[2] / 2)
    link_sim_mats = torch.zeros([links.shape[0], l1, l1]).cuda()

    for s in range(links.shape[0]):
        # print('links_sim_step', s, 'total step:', links.shape[0])
        adj = adjs[s, :, :]
        links_of_samlpe = links[s, :, :]
        N1 = torch.where(links_of_samlpe[:, :node_num])[1]  # 出发节点
        N2 = torch.where(links_of_samlpe[:, node_num:])[1]  # 到达节点
        link_sim_mat = torch.zeros([l1, l1]).cuda()

        for i in range(l1):
            v1 = N1[i]  # link1出发节点
            v2 = N2[i]  # link1达到节点
            adj_vector1 = adj[v1, :]
            adj_vector2 = adj[v2, :]
            for j in range(l1):
                if j > i:
                    v3 = N1[j]  # link2出发节点
                    v4 = N2[j]  # link2达到节点
                    adj_vector3 = adjs[s, v3, :]
                    adj_vector4 = adjs[s, v4, :]

                    s1 = torch.abs(adj_vector1-adj_vector3)
                    s2 = torch.abs(adj_vector2-adj_vector4)
                    s1 = torch.sum(s1)
                    s2 = torch.sum(s2)
                    sim = 2 ** (-s1) + 2 ** (-s2)

                    link_sim_mat[i, j] = sim
                    link_sim_mat[j, i] = sim

        link_sim_mats[s, :, :] = link_sim_mat
    return link_sim_mats


def links_to_triplets(links):
    # 输入链路序列和节点数量
    # 输出链路序列每个节点的三元组向量
    node_num = int(links.shape[2] / 2)
    Link_vectors1 = torch.zeros_like(links).cuda()
    # 所有样本的链路的闭集三元组特征向量
    Link_vectors2 = torch.zeros_like(links).cuda()
    # 所有样本的链路的开集三元组特征向量

    for i in range(links.shape[0]):
        # print('Triplets of node step:', i, 'total steps:', links.shape[0])
        links_of_sample = links[i, :, :]
        N1 = torch.where(links_of_sample[:, :node_num])[1] + 1  # 出发节点
        N2 = torch.where(links_of_sample[:, node_num:])[1] + 1  # 到达节点
        N3 = torch.unique(torch.cat((N1, N2)))  # 链路序列包含不同节点数量
        x1 = torch.zeros([node_num, node_num]).cuda()  # 闭集三元组向量
        x2 = torch.zeros([node_num, node_num]).cuda()  # 开集三元组向量
        for v1 in N3:
            V0 = torch.where(N2 == v1, N1, torch.zeros_like(N1))
            V2 = torch.where(N1 == v1, N2, torch.zeros_like(N1))

            if torch.where(V0)[0].shape[0] == 0 and torch.where(V0)[0].shape[0] >= 2:
                for v2 in V2[torch.where(V2)]:
                    x2[v1 - 1, v2 - 1] += 1

            if torch.where(V0)[0].shape[0] == 0 and torch.where(V0)[0].shape[0] >= 2:
                for v0 in V0[torch.where(V0)]:
                    x2[v1 - 1, v0 - 1] += 1

            if torch.where(V0)[0].shape[0] > 0 and torch.where(V0)[0].shape[0] > 0:
                x2[v1 - 1, V0[torch.where(V0)] - 1] += 1
                x2[v1 - 1, V2[torch.where(V2)] - 1] += 1
                for v2 in V2[torch.where(V2)]:
                    N4 = torch.where(N1 == v2, N2, torch.zeros_like(N1))
                    N5 = N4[torch.where(N4)]  # 节点v2的到达节点集
                    N6 = torch.where(N2 == v2, N1, torch.zeros_like(N1))
                    N7 = N6[torch.where(N6)]  # 节点v2的出发节点集

                    N8 = torch.cat((N5, N7))
                    if len(N8) > 0:
                        for v5 in N8:
                            if torch.where(V0 == v5)[0].shape[0] > 0:
                                # 节点v2的到达节点v5也是v1的出发节点v0时组成闭环
                                x1[v1 - 1, v5 - 1] += 1

        lv1 = torch.zeros_like(links_of_sample).cuda()
        lv2 = torch.zeros_like(links_of_sample).cuda()  # 单个样本中开集三元组向量

        for j in range(links_of_sample.shape[0]):
            lv1[j, :] = torch.cat((x1[N1[j] - 1, :], x1[N2[j] - 1, :]))
            lv2[j, :] = torch.cat((x2[N1[j] - 1, :], x2[N2[j] - 1, :]))

        Link_vectors1[i, :, :] = lv1
        Link_vectors2[i, :, :] = lv2

    return Link_vectors1, Link_vectors2


def links_laplace_matrces(output1, output2):
    samples = output1.shape[0]
    nodes = int(output1.shape[2])
    Lps = torch.zeros([samples, nodes, nodes]).cuda()
    for i in range(output1.shape[0]):
        Adj = torch.zeros([nodes, nodes]).cuda()
        for j in range(output1.shape[1]):
            v1 = torch.max(output1[i, j, :], dim=0)[1]
            v2 = torch.max(output2[i, j, :], dim=0)[1]
            Adj[v1, v2] += 1
            Adj[v2, v1] += 1
        # print(Adj)
        d = torch.sum(Adj, dim=1)
        d_sqrt = 1 / torch.sqrt(d)
        d_sqrt = torch.where(d > 0, d_sqrt, d)
        D_sqrt = torch.diag(d_sqrt)
        I = torch.ones_like(d)
        I = torch.where(d > 0, I, d)
        I = torch.diag(I)
        L1 = torch.mm(Adj, D_sqrt)
        L2 = torch.mm(D_sqrt, L1)
        Lps[i, :, :] = I - L2

    return Lps

# output1 = torch.rand([20, 20, 200])
# output2 = torch.rand([20, 20, 200])
# output3 = torch.rand([20, 20, 200])
# output4 = torch.rand([20, 20, 200])
# Lps1 = links_laplace_matrces(output1, output2)
# Lps2 = links_laplace_matrces(output3, output4)
# L1 = Lps1 - Lps2
# L2 = torch.norm(L1, dim=(1, 2))
# loss1 = torch.sum(L2)
# print(L1)
# print(L2)
# print(loss1)

# links = torch.zeros([5, 4, 8]).cuda()
# for i in range(links.shape[0]):
#     for j in range(links.shape[1]):
#         indx = torch.randint(0, int(links.shape[2]/2), [2])
#         links[i, j, indx[0]] = 1
#         links[i, j, indx[1]+int(links.shape[2]/2)] = 1
#
# print(links)
# print(links.shape)
#
# print(links_to_sim(links).shape)
