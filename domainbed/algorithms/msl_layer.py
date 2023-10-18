import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random

class Mahalanobis(nn.Module):
    def __init__(self, in_features, num, r=64):
        super(Mahalanobis, self).__init__()
        self.in_features = in_features
        self.out_features = num
        self.classnum = num
        self.margin = 0.1
        self.r = r

        self.bias = Parameter(torch.Tensor(num, in_features))
        self.weight = Parameter(torch.Tensor(num, r, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0))
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """
        shape of x is BxN, B is the batch size
        """
        B, N = x.shape
        h = x.unsqueeze(1).expand(B, self.classnum, N) - self.bias
        expanded_weight = self.weight.unsqueeze(0).expand(B, self.classnum, self.r, N)
        h = h.view(B*self.classnum, N).unsqueeze(2)
        expanded_weight = expanded_weight.reshape(B*self.classnum, self.r, N)

        s_r = torch.matmul(expanded_weight, h).squeeze()
        s = torch.square(torch.norm(s_r, dim=1, p=2))
        out = s.view(B, self.classnum)

        return out
