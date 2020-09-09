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
        output = torch.matmul(input, self.weight)
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
        support = torch.matmul(input, self.weight)
        if len(support.size()) == 3:
            output = torch.matmul(adj.view(1,adj.size()[0],adj.size()[1]), support)
        else:
            output = torch.matmul(adj, support)
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
        self.gc2 = GraphConvolution(nhid, nresourcees+1)
        self.gc3 = GraphConvolution(nhid, 1)
        self.c1 = ConvNet(nnodes, 1)
        self.c2 = ConvNet(nnodes, 1)
        self.dropout = dropout

    def forward(self, x, feat, adj):
        '''
        experiment with dropout
        '''
        x = torch.cat((x, feat), dim=-1)
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(self.c1.weight)
        x2 = self.gc2(x, adj)
        y = self.gc3(x, adj)

        if len(x2.size())==3:
            x2 = x2.permute(0,2,1)
            y = y.permute(0,2,1)
        else:
            x2 = x2.permute(1,0)
            y = y.permute(1,0)

        #print(x2.requires_grad)
        finx = self.c1(x2)
        finy = self.c2(y)

        finx = finx.view(finx.size()[:-1])
        finy = finy.view(finy.size()[:-1])
        #finy = self.c2(y).view(y.size()[:-1])


        return finx, F.softmax(finx, dim=0), finy

    def step(self, x, feat, adj):
        x = torch.FloatTensor(x)
        feat = torch.FloatTensor(feat)
        adj = torch.FloatTensor(adj)

        _, pi, v = self.forward(x, feat, adj)
        #pi, v = pi.detach().numpy().flatten(), v.detach().numpy()
        #return pi.detach().numpy().flatten(), v.detach().numpy()[0]
        return pi, v[0]

class GCNNode(nn.Module):

    def __init__(self, nfeat, nhid, ndegree, nnodes, dropout, nidx):
        super(GCNNode, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, ndegree)
        self.gc3 = GraphConvolution(nhid, 1)
        self.c1 = ConvNet(nnodes, 1)
        self.c2 = ConvNet(nnodes, 1)
        self.dropout = dropout

    def forward(self, x, feat, adj):
        '''
        experiment with dropout
        '''
        x = torch.cat((x, feat), dim=-1)
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x2 = self.gc2(x, adj)
        y = self.gc3(x, adj)

        if len(x2.size())==3:
            x2 = x2.permute(0,2,1)
            y = y.permute(0,2,1)
        else:
            x2 = x2.permute(1,0)
            y = y.permute(1,0)

        finx = self.c1(x2)
        finy = self.c2(y)


        finx = finx.view(finx.size()[:-1])
        finy = finy.view(finy.size()[:-1])
        #finy = self.c2(y).view(y.size()[:-1])

        return finx, F.softmax(finx, dim=0), finy

    def step(self, x, feat, adj):

        x = torch.FloatTensor(x)
        torch.FloatTensor(feat)
        adj = torch.FloatTensor(adj)

        _, pi, v = self.forward(x, feat, adj)
        #pi, v = pi.detach().numpy().flatten(), v.detach().numpy()
        #return pi.detach().numpy().flatten(), v.detach().numpy()[0]
        return pi, v[0]

class RepresentationFunc(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(RepresentationFunc, self).__init__()

        self.gc1 = GraphConvolution(nin, nhid)
        self.gc2 = GraphConvolution(nhid, nout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x,adj))
        return x

    def step(self, x, adj):
        x = torch.FloatTensor(x)
        adj = torch.FloatTensor(adj)
        return self.forward(x, adj)

class ScorePredictionFunc(nn.Module):
    def __init__(self, nfeat, nhid, nnodes, dropout):
        super(ScorePredictionFunc, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.c1 = ConvNet(nhid, 1)
        self.c2 = ConvNet(nnodes, 1)
        self.dropout = dropout

    def forward(self, x, feat, adj):
        '''
        experiment with dropout
        '''
        x = torch.cat((x, feat), dim=-1)
        x = F.relu(self.gc(x, adj))
        x2 = self.c1(x)

        if len(x2.size())==3:
            x2 = x2.permute(0,2,1)
        else:
            x2 = x2.permute(1,0)

        finx = F.relu(self.c2(x2))

        finx = finx.view(finx.size()[:-1])

        return finx

    def step(self, x, feat, adj):

        x = torch.FloatTensor(x)
        torch.FloatTensor(feat)
        adj = torch.FloatTensor(adj)

        return self.forward(x, feat, adj)
