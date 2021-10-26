import torch
import torch.utils.data
import numpy as np


def process_data_links(line, r, s):
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


def process_data_matrices(line, r, s):
    arr = []
    for data in line:
        data1 = data.strip('\n')
        # a, b, _ = data1.split(',')  # 这里不同的数据即要做不同的处理
        a, b, c, d = data1.split(' ')
        arr.append([a, b])
    # list->array
    arr_2 = np.array(arr, int)
    node_num = np.max(arr_2)
    # print("共变化了多少时间点：", len(arr_2))

    data = []  # 最后的数据存放的矩阵

    i, j = 0, r  # i表示窗口头部坐标，j表示窗口尾部坐标
    while j < len(arr_2):
        matric = np.eye(node_num).astype('int32')  # 这是每一份数据的矩阵，np.eye()表示增加了自环

        for k in range(i, j):
            # 每一个滑动窗口内连接的边都将矩阵相应位置变1
            matric[arr_2[k][0] - 1, arr_2[k][1] - 1] += 1
            matric[arr_2[k][1] - 1, arr_2[k][0] - 1] += 1
        data.append(matric)  # 加入最后存放数据的矩阵
        i, j = i + s, j + s  # 窗口滑动步长s

    # 处理最后不足窗口大小的时间点数量
    matric = np.eye(node_num).astype('int32')
    for k in range(i, len(arr_2)):
        matric[arr_2[k][0] - 1, arr_2[k][1] - 1] += 1
        matric[arr_2[k][1] - 1, arr_2[k][0] - 1] += 1
    data.append(matric)
    data = np.array(data)

    return data, node_num


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
                 output_size=1,  # 单个样本输出link序列的数量
                 use_snapshot=False, ):  # 是否使用邻接矩阵表示链路序列):

        self.r = r
        self.s = s
        self.link = link
        self.input_size = input_size
        self.output_size = output_size
        self.use_snapshot = use_snapshot
        self.samples, self.labels = self.__getsamples()

    def __getsamples(self):

        if self.use_snapshot:
            data, self.nodes = process_data_matrices(self.link, self.r, self.s)
            self.len = data.shape[0]
            self.sample_num = int(np.floor((self.len - self.input_size) / self.output_size))
            X = torch.zeros((self.sample_num, self.input_size, self.nodes, self.nodes))
            Y = torch.zeros((self.sample_num, self.output_size, self.nodes, self.nodes))

            for i in range(self.sample_num):
                start = i
                end = i + self.input_size
                X[i, :, :, :] = torch.from_numpy(data[start:end, :, :])
                Y[i, :, :, :] = torch.from_numpy(data[end:end + self.output_size, :, :])
        else:
            data, self.nodes = process_data_links(self.link, self.r, self.s)
            self.len = data.shape[0]
            self.sample_num = int(np.floor((self.len - self.input_size) / self.output_size))
            X = torch.zeros((self.sample_num, self.input_size * self.r, self.nodes * 2))
            Y = torch.zeros((self.sample_num, self.output_size * self.r, self.nodes * 2))

            for i in range(self.sample_num):
                start = i
                end = i + self.input_size

                input_links = data[start:end, :, :]
                output_links = data[end:end + self.output_size, :, :]

                input_links = input_links.reshape(input_links.shape[0] * input_links.shape[1], input_links.shape[2])
                # 将输入的若干个link序列拼接成一个（input_size*self.r,2）
                output_links = output_links.reshape(output_links.shape[0] * output_links.shape[1],
                                                    output_links.shape[2])
                # output_links = output_links.reshape(1, output_links.shape[0] * 2)

                for j in range(input_links.shape[0]):
                    v1 = torch.zeros(self.nodes)
                    v2 = torch.zeros(self.nodes)
                    v1[input_links[j, 0] - 1] = 1
                    v2[input_links[j, 1] - 1] = 1
                    link_vector = torch.cat([v1, v2], dim=0).float()  # 用节点的one-hot向量拼接成原始link向量
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


def getdata(train, us):
    if train:  # 这样不太好，应该在处理数据的类里面返回训练集和测试集
        filepath = '../data/rawdata/enron_train.edges'
    else:
        filepath = '../data/rawdata/enron_test.edges'
    f = open(filepath)
    link = f.readlines()
    return DNDataset(link=link, r=500, s=500, input_size=1, output_size=1, use_snapshot=us)


def getdata2(train, us, r, s):
    if train:  # 这样不太好，应该在处理数据的类里面返回训练集和测试集
        filepath = '../data/rawdata/enron_train.edges'
    else:
        filepath = '../data/rawdata/enron_test.edges'
    f = open(filepath)
    link = f.readlines()
    return DNDataset(link=link, r=r, s=s, input_size=1, output_size=1, use_snapshot=us)
# data = getdata(True, False)
# print(data)
# print(data[0][0].shape)
# print(data[0][1].shape)
