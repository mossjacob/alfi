import operator
import torch

from functools import reduce
from torch.distributions import MultivariateNormal
from torch.nn.functional import softplus

from lafomo.nn import SimpleBlock1d, SimpleBlock2d
from lafomo.models import LFM


class NeuralOperator(LFM):
    def __init__(self, block_dim, in_channels, out_channels, modes, width, num_layers=4):
        super().__init__()
        self.num_outputs = 1
        if block_dim == 1:
            self.conv = SimpleBlock1d(in_channels, out_channels, modes, width, num_layers=num_layers)
        elif block_dim == 2:
            self.conv = SimpleBlock2d(in_channels, out_channels, modes, modes, width, num_layers=num_layers)
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

    def predict_f(self, tx_predict):
        with torch.no_grad():
            y_pred, params_out = self(tx_predict)
            p_y_mean = y_pred[..., 0]
            p_y_var = softplus(y_pred[..., 1])
            p_y_pred = MultivariateNormal(p_y_mean, p_y_var)
            return p_y_pred, params_out