import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Linear, Conv1d
from .spectral_conv import SpectralConv1d, SpectralConv2d
from alfi.models.mlp import MLP


"""
Neural network layer for neural operator.
"""
class NNBlock1d(Module):
    def __init__(self, in_channels, out_channels, modes, width, num_layers=4, **kwargs):
        super(NNBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.fc0 = Linear(in_channels, self.width)
        self.fc0_parameters = Linear((1 + self.modes1 * num_layers*2)* self.width, 50)
        self.fc1_parameters = Linear(50, 5*3)
        self.spectral_layers = ModuleList([
            MLP(self.width, self.width, latent_dim=self.modes1, num_hidden_layers=0)
            for _ in range(num_layers)
        ])
        self.weight_layers = ModuleList([
            Conv1d(self.width, self.width, 1)
            for _ in range(num_layers)
        ])

        self.fc1 = Linear(self.width, 128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.fc0(x)

        for i in range(self.num_layers):
            conv = self.spectral_layers[i]
            w = self.weight_layers[i]

            x1 = conv(x)
            x2 = w(x.permute(0, 2, 1)).permute(0, 2, 1)

            x = x1 + x2
            if i < (self.num_layers - 1):
                x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class SimpleBlock2d(Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width, num_layers=4, params=True):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=in_channels)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.params = params
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.fc0 = Linear(in_channels, self.width)
        if params:
            self.fc0_parameters = Linear(self.width * self.modes1 * self.modes2 * num_layers**2 * 2, 200)
            self.fc1_parameters = Linear(200, 5)
        self.spectral_layers = ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(num_layers)
        ])
        self.weight_layers = ModuleList([
            Conv1d(self.width, self.width, 1)
            for _ in range(num_layers)
        ])

        self.fc1 = Linear(self.width, 128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        out_fts = torch.zeros(batchsize, self.width, self.modes1*self.num_layers, self.modes2*self.num_layers,
                              dtype=torch.complex64, device=x.device)

        for i in range(self.num_layers):
            conv = self.spectral_layers[i]
            w = self.weight_layers[i]
            x1, out_ft = conv(x)
            out_fts[..., i*self.modes1:(i+1)*self.modes1, i*self.modes2:(i+1)*self.modes2] = \
                out_ft[..., :self.modes1, :self.modes2]
            x2 = w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            if i < (self.num_layers - 1):
                x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        if self.params:
            out_fts = torch.stack([out_fts.real, out_fts.imag], dim=-1)
            params = self.fc0_parameters(out_fts.reshape(batchsize, -1))
            params = F.relu(params)
            params = self.fc1_parameters(params)
            return x, params

        return x
