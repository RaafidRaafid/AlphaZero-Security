import unittest
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace


x = torch.rand(16,10)
adj = torch.rand(10,10)

print(torch.max(x, dim = 1))
