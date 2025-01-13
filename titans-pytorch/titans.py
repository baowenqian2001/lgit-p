from __future__ import annotations

import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad_and_value 

from einops import rearrange
from einops.layers.torch import Rearrange

from einops import rearrange
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(Linear, bias=False) # 避免总是需要输入相同的参数

# functions
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d 

def round_down_multiple(seq, mult):
    return seq // mult * mult

# classes
class MLP(Module):
    def __init__(
            self, 
            dim,
            depth
    ):
        super().__init__()
        # 确保父类初始化
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim,dim)) for _ in range(depth)])

    def forward(
            self,
            x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.silu(x) # sigmoid linear unit 缩写
            
            x = x @ weight




