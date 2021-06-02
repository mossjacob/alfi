import operator
import torch

from functools import reduce
from torch.distributions import Normal
from torch.nn.functional import softplus

from alfi.nn import SimpleBlock1d, SimpleBlock2d
from alfi.models import LFM
from alfi.models import NeuralOperator


class RecurrentNeuralOperator(LFM):
    def __init__(self, block_dim, in_channels, out_channels, modes, width, num_layers=4):
        super().__init__()
        self.params = False
        self.operator = NeuralOperator(block_dim, in_channels, out_channels, modes, width,
                                       num_layers=num_layers, params=False)

    def forward(self, x, step=1):
        """
        x shape (batch, space, times, channels)
        """
        batch_size, T, S, _ = x.shape
        # Last dimension is time
        pred = None
        for t in range(0, T, step):
            y_pred = self.operator(x[:, t, ...]).unsqueeze(-1)  # (batch, space, channels, 1)
            if t == 0:
                pred = y_pred
            else:
                pred = torch.cat((pred, y_pred), -1)

            # x = torch.cat((x[..., step:], y_pred), dim=-1)

        return pred.permute(0, 3, 1, 2)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

    def predict_f(self, tx_predict):
        with torch.no_grad():
            y_pred = self(tx_predict)
            p_y_mean = y_pred[..., 0]
            p_y_var = softplus(y_pred[..., 1])
            p_y_pred = Normal(p_y_mean, p_y_var)
            return p_y_pred

    def save(self, filepath):
        torch.save(self.state_dict(), str(filepath) + 'lfo.pt')
