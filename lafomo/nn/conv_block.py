import torch.nn.functional as F

from torch.nn import Module, Linear, Conv1d
from .spectral_conv import SpectralConv1d, SpectralConv2d


"""
Authors: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A. and Anandkumar, A., 2020. 
Fourier neural operator for parametric partial differential equations. 
arXiv preprint arXiv:2010.08895.
"""
class SimpleBlock1d(Module):
    def __init__(self, in_channels, modes, width):
        super(SimpleBlock1d, self).__init__()

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
        self.fc0 = Linear(in_channels, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = Conv1d(self.width, self.width, 1)
        self.w1 = Conv1d(self.width, self.width, 1)
        self.w2 = Conv1d(self.width, self.width, 1)
        self.w3 = Conv1d(self.width, self.width, 1)


        self.fc1 = Linear(self.width, 128)
        self.fc2 = Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleBlock2d(Module):
    def __init__(self, in_channels, modes1, modes2,  width):
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

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = Linear(in_channels, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc0_parameters = Linear(self.width * self.modes1 * self.modes2, 100)
        self.fc1_parameters = Linear(100, 4)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = Conv1d(self.width, self.width, 1)
        self.w1 = Conv1d(self.width, self.width, 1)
        self.w2 = Conv1d(self.width, self.width, 1)
        self.w3 = Conv1d(self.width, self.width, 1)


        self.fc1 = Linear(self.width, 128)
        self.fc2 = Linear(128, 2)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        out_fts = 0
        x1, out_ft = self.conv0(x)
        # out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1,
        out_fts += out_ft
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1, out_ft = self.conv1(x)
        out_fts += out_ft
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1, out_ft = self.conv2(x)
        out_fts += out_ft
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1, out_ft = self.conv3(x)
        out_fts += out_ft
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        params = self.fc0_parameters(out_fts[:, :, :self.modes1, :self.modes2].real.reshape(batchsize, -1))
        params = self.fc1_parameters(params)
        # print('params', params.shape)
        # out_ft[:, :, -self.modes1:, :self.modes2] = \

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x, params
