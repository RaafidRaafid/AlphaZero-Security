import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils import *

class denseNet(Module):

    def __init__(self, in_feat, out_feat, bias=True):
        super(denseNet, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self, in_features, out_features, activation=None, bnorm=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        # if len(x[0].shape)==3:
        #     print("mamoooo", x[0].shape, x[1].shape)
        laplacian = x[1]
        if len(x[0].shape) == 2:
            x_hat = torch.matmul(laplacian, x[0])
        else:
            x_hat = torch.bmm(laplacian, x[0])
        idx = x[2]

        # if idx==0:
        #     print("age", x_hat)
        x = self.fc(x_hat)
        # if idx==0:
        #     print("pore", x)

        if self.bnorm:
            if len(x.shape) == 3:
                x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return [x,laplacian,idx+1]

class RepresentationFunc(nn.Module):
    def __init__(self, in_features, out_features, adj, filters=[128, 128], bnorm=False, n_hidden=0, dropout=0.2, noGCN = False, debugging = False):
        super(RepresentationFunc, self).__init__()

        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                # activation=nn.ReLU(inplace=True),
                                                activation = nn.Tanh(),
                                                bnorm=bnorm) for layer, f in enumerate(filters)]))
        if noGCN:
            self.laplacian = torch.eye(adj.shape[0])
        else:
            self.laplacian = self.laplacian_batch(adj)

        self.fc = nn.Linear(filters[-1], out_features)
        self.activation = nn.ReLU(inplace = True)

    def laplacian_batch(self, A):
        A = torch.FloatTensor(A)
        N = A.shape[0]
        A_hat = A
        I = torch.eye(N)
        I = 2 * I
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def forward(self, x, feat):

        if len(x.shape)==3:
            self.t_laplacian = [self.laplacian.unsqueeze(dim=0)]*x.shape[0]
            self.t_laplacian = torch.cat(self.t_laplacian)
        else:
            self.t_laplacian = self.laplacian

        x = torch.cat((x, feat), dim=-1)
        # print("magix", x)

        x = self.gconv([x, self.t_laplacian, 0])[0]
        x = self.fc(x)
        x = self.activation(x)
        return x

    def step(self, x, feat):
        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        return self.forward(x, feat)


class PredictionNN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self, in_features, out_features, n_hidden=64, dropout=0.2, debugging = False):
        super(PredictionNN, self).__init__()

        self.debugging = debugging

        # Fully connected layers
        fcPolicy = []
        fcQ = []
        if dropout > 0:
            fcPolicy.append(nn.Dropout(p=dropout))
            fcQ.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fcPolicy.append(nn.Linear(in_features, n_hidden))
            fcPolicy.append(nn.ReLU(inplace=True))
            fcQ.append(nn.Linear(in_features, n_hidden))
            fcQ.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fcPolicy.append(nn.Dropout(p=dropout))
                fcQ.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fcPolicy.append(nn.Linear(n_last, out_features))
        fcQ.append(nn.Linear(n_last, 1))
        self.fcPolicy = nn.Sequential(*fcPolicy)
        self.fcQ = nn.Sequential(*fcQ)

    def laplacian_batch(self, A):
        A = torch.FloatTensor(A)
        N = A.shape[0]
        A_hat = A
        I = torch.eye(N)
        I = 2 * I
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def forward(self, x):

        # x = torch.mean(x, dim=-2)  # max pooling over nodes (usually performs better than average)
        x = torch.max(x, dim=-2)[0]  # max pooling over nodes (usually performs better than average)
        if self.debugging:
            print(x)
        Policy = self.fcPolicy(x)
        # if self.debugging:
        #     print(F.softmax(Policy, dim=0))
        Q = self.fcQ(x)
        return Policy, F.softmax(Policy, dim=0), Q

    def step(self, x):

        x = torch.FloatTensor(x)

        _, pi, v = self.forward(x)
        return pi, v[0]

class ScorePredictionNN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self, in_features, n_hidden=64, dropout=0.2, debugging = False):
        super(ScorePredictionNN, self).__init__()

        self.debugging = debugging

        # Fully connected layers
        fc =[]
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(in_features, n_hidden))
            fc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, 1))
        self.fc = nn.Sequential(*fc)

    def laplacian_batch(self, A):
        A = torch.FloatTensor(A)
        N = A.shape[0]
        A_hat = A
        I = torch.eye(N)
        I = 2 * I
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def forward(self, x):

        # x = torch.mean(x, dim=-2)  # max pooling over nodes (usually performs better than average)
        x = torch.max(x, dim=-2)[0]  # max pooling over nodes (usually performs better than average)
        if self.debugging:
            print(x)
        score = self.fc(x)
        return score

    def step(self, x):

        x = torch.FloatTensor(x)

        return self.forward(x)
