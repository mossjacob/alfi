import operator

from torch.nn import Module
from functools import reduce
from lafomo.nn import SimpleBlock1d, SimpleBlock2d


class OperatorModel(Module):
    def __init__(self, block_dim, modes, width):
        super().__init__()
        if block_dim == 1:
            self.conv = SimpleBlock1d(modes, width)
        elif block_dim == 2:
            self.conv = SimpleBlock2d(modes, modes, width)
        else:
            pass

    def forward(self, x):
        x = self.conv(x)
        return x#.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

