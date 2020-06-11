# -*- coding: utf-8 -*-
# date: 2018-11-29 20:14
import torch

import torch.nn as nn
from distiller.modules import *


class Std(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Std, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        return torch.std(x, *self.args, **self.kwargs)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.Mean = Mean(dim=-1, keepdim=True)
        self.Std = Std(dim=-1, keepdim=True)
        self.eltwisemul1 = EltwiseMult()
        self.eltwisesub1 = EltwiseSub()
        self.eltwisediv = EltwiseDiv()
        self.eltwiseadd = EltwiseAdd()

    def forward(self, x):
        # mean = x.mean(dim=-1, keepdim=True)
        mean = self.Mean(x)
        # std = x.std(dim=-1, keepdim=True)
        std = self.Std(x)
        return self.eltwisediv(self.eltwisemul1(self.a_2, (self.eltwisesub1(x - mean))),
                               self.eltwiseadd((std + self.eps), self.b_2))
