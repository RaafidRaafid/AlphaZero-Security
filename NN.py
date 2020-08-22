import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils import *

class ConvNet(Module):

    def __init__(self, in_feat, out_feat, bias=True):
        super(ConvNet, self).__init__()
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
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution(Module):

    def __init__(self, in_feat, out_feat, bias=True):
        super(GraphConvolution, self).__init__()
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

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNBoard(nn.Module):
    def __init__(self, nfeat, nhid, nresourcees, nnodes, dropout):
        super(GCNBoard, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nresourcees)
        self.gc3 = GraphConvolution(nhid, 1)
        self.c1 = ConvNet(nnodes, 1)
        self.c2 = ConvNet(nnodes, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        '''
        experiment with dropout
        '''
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(self.c1.weight)
        x2 = self.gc2(x, adj)
        y = self.gc3(x, adj)

        x2 = torch.transpose(x2,0,1)
        y = torch.transpose(y,0,1)

        #print(x2.requires_grad)
        finx = self.c1(x2)
        finy = self.c2(y).view(-1)


        return F.softmax(finx, dim=1), finy

    def step(self, x, adj):
        x = torch.FloatTensor(x)
        adj = torch.FloatTensor(adj)

        pi, v = self.forward(x,adj)
        #pi, v = pi.detach().numpy().flatten(), v.detach().numpy()
        #return pi.detach().numpy().flatten(), v.detach().numpy()[0]
        return pi.flatten(), v[0]

class GCNNode(nn.Module):

    def __init__(self, nfeat, nhid, ndegree, nnodes, dropout, nidx):
        super(GCNNode, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, ndegree)
        self.gc3 = GraphConvolution(nhid, 1)
        self.c1 = ConvNet(nnodes, 1)
        self.c2 = ConvNet(nnodes, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        '''
        experiment with dropout
        '''

        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x2 = self.gc2(x, adj)
        y = self.gc3(x, adj)

        x2 = torch.transpose(x2,0,1)
        y = torch.transpose(y,0,1)

        finx = self.c1(x2)
        finy = self.c2(y).view(-1)

        return F.softmax(finx, dim=1), finy

    def step(self, x, adj):

        x = torch.FloatTensor(x)
        adj = torch.FloatTensor(adj)

        pi, v = self.forward(x,adj)
        #pi, v = pi.detach().numpy().flatten(), v.detach().numpy()
        #return pi.detach().numpy().flatten(), v.detach().numpy()[0]
        return pi.flatten(), v[0]
