import unittest
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace


x = torch.rand(16,10,3)
adj = torch.rand(10,10)

wow = torch.mm(adj.view(1,adj.size()[0],adj.size()[1]), x)
print(x,adj,wow)
