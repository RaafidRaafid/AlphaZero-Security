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

class GCNBoard(nn.Module):
    def __init__(self, nfeat, nhid, nresourcees, dropout):
        super(GCNBoard, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.cPolicy = denseNet(nhid, nresourcees+1)
        self.cQ = denseNet(nhid, 1)
        self.dropout = dropout

    def forward(self, x, feat, edge_index, is_training = True):
        '''
        experiment with dropout
        '''
        x = torch.cat((x, feat), dim=-1)
        x = x.squeeze()

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = torch.max(x, dim = len(x.shape)-2)[0]

        finx = self.cPolicy(x)
        finy = self.cQ(x)

        return finx, F.softmax(finx, dim=0), finy

    def step(self, x, feat, edge_index, is_training = True):
        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        edge_index = torch.LongTensor(edge_index)

        _, pi, v = self.forward(x, feat, edge_index, is_training)
        return pi, v[0]

class GCNNode(nn.Module):

    def __init__(self, nfeat, nhid, ndegree, dropout):
        super(GCNNode, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.cPolicy = denseNet(nhid, ndegree)
        self.cQ = denseNet(nhid, 1)
        self.dropout = dropout

    def forward(self, x, feat, edge_index, is_training = True):
        '''
        experiment with dropout
        '''

        x = torch.cat((x, feat), dim=-1)
        x = x.squeeze()

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = torch.max(x, dim = len(x.shape)-2)[0]

        finx = self.cPolicy(x)
        finy = self.cQ(x)

        return finx, F.softmax(finx, dim=0), finy

    def step(self, x, feat, edge_index, is_training = True):

        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        edge_index = torch.LongTensor(edge_index)

        _, pi, v = self.forward(x, feat, edge_index, is_training)
        #pi, v = pi.detach().numpy().flatten(), v.detach().numpy()
        #return pi.detach().numpy().flatten(), v.detach().numpy()[0]
        return pi, v[0]

class RepresentationFunc(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(RepresentationFunc, self).__init__()

        self.gc1 = GCNConv(nin, nhid)
        self.gc2 = GCNConv(nhid, nout)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gc1(x, edge_index))
        x = torch.tanh(self.gc2(x,edge_index))
        return x

    def step(self, x, edge_index):
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        return self.forward(x, edge_index)

class ScorePredictionFunc(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(ScorePredictionFunc, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.c1 = denseNet(nhid, nhid2)
        self.c2 = denseNet(nhid2, 1)
        self.dropout = dropout

    def forward(self, x, feat, edge_index, is_training = True):
        '''
        experiment with dropout
        '''
        x = torch.cat((x, feat), dim=-1)
        x = x.squeeze()

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training = is_training)

        x = torch.max(x, dim = len(x.shape)-2)[0]

        x = torch.tanh(self.c1(x))
        x = self.c2(x)

        return x

    def step(self, x, feat, edge_index, is_training = True):

        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        edge_index = torch.LongTensor(edge_index)

        return self.forward(x, feat, edge_index, is_training)
