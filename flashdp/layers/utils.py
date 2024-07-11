import torch.nn as nn

class DPParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, requires_dp=True):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.requires_dp = requires_dp
        return t