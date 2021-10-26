import torch
import torch.utils.data
import numpy as np
from data.clustering_encoding import ClusteringEncode


def process_data(line, r, s):
    # 输入链路序列line, 每个网络包含节点数量r,跳步数量s
    # 输出切分好的链路序列和节点数量
    arr = []
    for data in line:
        data1 = data.strip('\n')
        # a, b, _ = data1.split('，')  # 这里不同的数据即要做不同的处理
        a, b, c, d = data1.split(' ')
        arr.append([a, b])
    # list->array
    arr_2 = np.array(arr, int)
    node_num = np.max(arr_2)
    # print("共变化了多少时间点：", len(arr_2))

    data = []  # 最后的数据存放的矩阵

    i, j = 0, r  # i表示窗口头部坐标，j表示窗口尾部坐标
    while j < len(arr_2):
        links = arr_2[i:j, :]
        data.append(links)  # 加入最后存放数据的矩阵
        i, j = i + s, j + s  # 窗口滑动步长s

    # 处理最后不足窗口大小的时间点数量
    links = arr_2[len(arr_2) - r:len(arr_2), :]
    data.append(links)
    data = np.array(data)
    # print(data.shape)
    return data, node_num


def links_to_triplets(links, node_num):
    # 输入链路序列和节点数量
    # 输出链路序列每个节点的三元组向量
    N1 = links[:, 0]  # 出发节点
    N2 = links[:, 1]  # 到达节点
    N3 = np.unique(links)  # 链路序列包含不同节点数量
    x1 = np.zeros([node_num, node_num])  # 闭集三元组向量
    x2 = np.zeros([node_num, node_num])  # 开集三元组向量

    for v1 in N3:
        V0 = N1[np.where(N2 == v1)]
        V2 = N2[np.where(N1 == v1)]
        if len(V0) == 0 and len(V2) >= 2:
            for v2 in V2:
                x2[v1 - 1, v2 - 1] += 1

        if len(V2) == 0 and len(V0) >= 2:
            for v0 in V0:
                x2[v1 - 1, v0 - 1] += 1

        if len(V0) > 0 and len(V2) > 0:
            x2[v1 - 1, V0 - 1] += 1
            x2[v1 - 1, V2 - 1] += 1
            for v2 in V2:
                N4 = N2[np.where(N1 == v2)]
                N5 = np.intersect1d(V0, N3)
                if len(N5) > 0:
                    for v4 in N4:
                        x1[v1 - 1, v4 - 1] += 1

    return x1, x2


class DNDataset(torch.utils.data.Dataset):
    """ 动态网络数据集

    Returns:
        [sample, label]
    """

    def __init__(self,
                 r=5,  # 一个link序列包含的link数量
                 s=5,  # 每次增加链路数量
                 link=None,  # 原始数据集的链路序列
                 input_size=1,  # 单个样本输入link序列的数量
                 output_size=1, ):  # 单个样本输出link序列的数量

        self.r = r
        self.s = s
        self.link = link
        self.input_size = input_size
        self.output_size = output_size

        data, self.nodes = process_data(self.link, self.r, self.s)
        self.len = data.shape[0]

        self.sample_num = int(np.floor((self.len - input_size) / output_size))
        self.samples, self.labels = self.__getsamples(data)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.input_size * self.r, self.nodes * 2))
        Y = torch.zeros((self.sample_num, self.output_size * self.r, self.nodes * 2))

        for i in range(self.sample_num):
            start = i
            end = i + self.input_size

            input_links = data[start:end, :, :]
            output_links = data[end:end + self.output_size, :, :]

            input_links = input_links.reshape(input_links.shape[0] * input_links.shape[1], input_links.shape[2])
            # 将输入的若干个link序列拼接成一个（input_size*self.r,2）
            output_links = output_links.reshape(output_links.shape[0] * output_links.shape[1], output_links.shape[2])
            # output_links = output_links.reshape(1, output_links.shape[0] * 2)

            T1, T2 = links_to_triplets(input_links, self.nodes)
            # 根据所有输入的链路序列构建每个节点的三元组向量

            for j in range(input_links.shape[0]):
                v1 = torch.zeros(self.nodes)
                v2 = torch.zeros(self.nodes)
                t1 = T1[input_links[j, 0] - 1, :]  # 输入的第j条链路中第1个节点的闭集三元组向量
                t2 = T2[input_links[j, 1] - 1, :]  # 输入的第j条链路中第2个节点的开三元组向量
                t1 = torch.from_numpy(t1).float()  # 转为浮点tensor
                t2 = torch.from_numpy(t2).float()  # 转为浮点tensor
                c = ClusteringEncode(node_num=self.nodes)
                # y1 = c(t1)
                hv1 = torch.cat([c(t1), c(t2)], dim=0)
                # 将两者三元组向量输入嵌入网络，并拼接成长向量（1, 2*node_num）
                # .detach不修改嵌入中的参数
                v1[input_links[j, 0] - 1] = 1
                v2[input_links[j, 1] - 1] = 1
                link_vector = torch.cat([v1, v2], dim=0).float()
                # 用节点的one-hot向量拼接成原始link向量
                X[i, j, :] = link_vector

            for j in range(output_links.shape[0]):
                v1 = torch.zeros(self.nodes)
                v2 = torch.zeros(self.nodes)
                v1[output_links[j, 0] - 1] = 1
                v2[output_links[j, 1] - 1] = 1
                link_vector = torch.cat([v1, v2], dim=0).float()
                # 用节点的one-hot向量拼接成原始link向量
                Y[i, j, :] = link_vector

        return X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        # sample = [self.samples[idx], self.labels[idx, :, :]]
        # return sample
        return self.samples[idx, :, :], self.labels[idx, :, :]


def getdata(train):
    if train:   # 这样不太好，应该在处理数据的类里面返回训练集和测试集
        filepath = '../data/rawdata/enron_test.edges'
    else:
        filepath = '../data/rawdata/test.edges'
    f = open(filepath)
    link = f.readlines()
    return DNDataset(link=link, r=500, s=500, input_size=1, output_size=1)


data = getdata(True)
print(data)