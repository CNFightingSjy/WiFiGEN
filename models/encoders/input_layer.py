import numpy as np
from torch import nn

class Mapping_layer(nn.Module):
    def __init__(self, in_num, out_num):
        super().__init__()
        self.in_num = in_num
        self.out_num = out_num
        modules = [nn.Linear(in_num, 19 * 19), nn.Linear(19 * 19, 16 * 16), nn.Linear(16 * 16, out_num * out_num)]
        self.mapping_layer = nn.Sequential(*modules)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # x = x.flatten()
        # print(x.dtype)
        x = x.float()
        # print(x.dtype)
        x = self.mapping_layer(x)
        # print(x.shape)
        x = x.view(x.shape[0], 1, self.out_num, self.out_num)
        return x