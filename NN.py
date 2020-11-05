import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
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

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class RepresentationFunc(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(RepresentationFunc, self).__init__()

        self.gc = []
        for layer, f in enumerate(nhid):
            self.gc.append(GCNConv(in_channels = nin if layer == 0 else nhid[layer-1], out_channels = f))

        self.fc = nn.Linear(nhid[-1], nout)
        self.activation = nn.ReLU(inplace = True)

    def forward(self, x, feat, edge_index):
        x = torch.cat((x, feat), dim=-1)
        x = torch.squeeze(x)

        for i in range(len(self.gc)):
            x = self.gc[i](x, edge_index)
            x = self.activation(x)

        x = self.fc(x)
        return self.activation(x)

    def step(self, x, feat, edge_index):
        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        edge_index = torch.LongTensor(edge_index)
        return self.forward(x, feat, edge_index)


class PredictionNN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nout, dropout):
        super(PredictionNN, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid2)
        self.fc3_1 = nn.Linear(nhid2, nout)
        self.fc3_2 = nn.Linear(nhid2, 1)
        self.activation = nn.ReLU(inplace = True)
        self.dropout = dropout

    def forward(self, x, is_training = True):
        '''
        experiment with dropout
        '''
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = torch.max(x, dim=-2)[0]

        logits = self.fc3_1(x)
        value = self.fc3_2(x)

        return logits, F.softmax(logits, dim=0), value

    def step(self, x, is_training = True):
        x = torch.FloatTensor(x)

        _, pi, v = self.forward(x, is_training)
        return pi, v[0]

class ScorePredictionNN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(ScorePredictionNN, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid2)
        self.fc3 = nn.Linear(nhid2, 1)
        self.activation = nn.ReLU(inplace = True)
        self.dropout = dropout

    def forward(self, x, is_training = True):
        '''
        experiment with dropout
        '''
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = torch.max(x, dim=-2)[0]

        value = self.fc3(x)

        return value

    def step(self, x, is_training = True):
        x = torch.FloatTensor(x)

        return self.forward(x, is_training)[0]
