import operator
import torch

from functools import reduce
from torch.distributions import Normal
from torch.nn.functional import softplus

from lafomo.nn import SimpleBlock1d, SimpleBlock2d
from lafomo.models import LFM
from lafomo.models import NeuralOperator


class RecurrentNeuralOperator(LFM):
    def __init__(self, block_dim, in_channels, out_channels, modes, width, num_layers=4):
        super().__init__()
        self.operator = NeuralOperator(block_dim, in_channels, out_channels, modes, width,
                                       num_layers=num_layers, params=False)

    def forward(self, x, T=1, step=1):
        # Last dimension is time
        pred = torch.empty_like(x)
        for t in range(0, T, step):
            y_pred = self.operator(x)
            print('pred', y_pred.shape)
            if t == 0:
                pred = y_pred
            else:
                pred = torch.cat((pred, y_pred), -1)

            x = torch.cat((x[..., step:], y_pred), dim=-1)

        return pred

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
            p_y_pred = Normal(p_y_mean, p_y_var)
            return p_y_pred, params_out

    def save(self, filepath):
        torch.save(self.state_dict(), str(filepath) + 'lfo.pt')
