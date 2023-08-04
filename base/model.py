
import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        input = F.dropout(input, self.dropout, self.training)

        D = torch.diag((adj > 0.).int().sum(dim=1))
        #print(D)

        support = torch.mm(input, self.weight)
        I = torch.zeros_like(adj).fill_diagonal_(1.)
        #print(adj)
        #print(  torch.matmul( torch.matmul((D ** 0.5), (adj + I)), (D ** 0.5))  )

        DD = torch.diag(D.diagonal() ** -0.5)

        A_tilda = DD @ (adj + I) @ DD

        #A_tilda = adj
        output = torch.spmm(A_tilda, support)

        return output + self.bias


class MBSClassifier(nn.Module):
    def __init__(self, nfeat, n_out, dropout):
        super(MBSClassifier, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 10, dropout)

        self.gc2 = GraphConvolution(10, 5, dropout)

        self.fc = fc = nn.Linear(in_features=5, out_features=2)

        self.fc1 = fc = nn.Linear(in_features=5, out_features=5)
        self.fc2 = fc = nn.Linear(in_features=5, out_features=2)

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))

        x = self.gc2(x, adj)

        z = x.mean(dim=0)
        z = F.relu(z)

        z2 = torch.relu(self.fc1(z))
        #y = self.fc2(z)

        y = torch.sigmoid(self.fc(z2))

        return z, y


class MBSScore(nn.Module):
    def __init__(self, nfeat, n_out, dropout):
        super(MBSScore, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat, dropout)

        self.gc2 = GraphConvolution(nfeat, 1, dropout)



    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))

        x = self.gc2(x, adj)

        return x