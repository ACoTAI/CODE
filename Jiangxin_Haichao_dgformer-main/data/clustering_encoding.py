import torch.nn as nn


class ClusteringEncode(nn.Module):

    def __init__(self, node_num):
        super().__init__()
        self.node_num = node_num
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=self.node_num, out_features=100, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100, bias=True),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=self.node_num, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.hidden3(fc2)
        return output
