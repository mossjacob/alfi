import operator
import torch

from functools import reduce
from torch.nn import Module, Linear
from torch.distributions import Normal

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


class MuSigmaEncoder(Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = Linear(r_dim, r_dim)
        self.hidden_to_mu = Linear(r_dim, z_dim)
        self.hidden_to_sigma = Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class NeuralLFM(LFM):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, block_dim, in_channels, modes, width, r_dim, z_dim):
        super(NeuralLFM, self).__init__()
        self.num_outputs = 1
        self.x_dim = 1
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.block_dim = block_dim
        # Initialize networks
        self.xy_to_r = NeuralOperator(block_dim, in_channels + 1, r_dim, modes, width, num_layers=2)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = NeuralOperator(block_dim, in_channels + z_dim, 2, modes, width, num_layers=2)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, s, r_dim)
        """
        dim = list(range(1, r_i.ndim-1))
        return torch.mean(r_i, dim=dim)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        if self.block_dim == 1:
            batch_size, num_points, _ = x.size()
        else:
            batch_size, num_points, _, _ = x.size()

        # Flatten tensors, as encoder expects one dimensional inputs
        # x_flat = x.view(batch_size * num_points, self.x_dim)
        # y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        xy = torch.cat([x, y], dim=-1)
        # print('xy', xy.shape)
        r_i, params_i = self.xy_to_r(xy)

        # Reshape tensors into batches
        # r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # print('agg', r.shape)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample().unsqueeze(1)
            # Get parameters of output distribution
            w1 = x_target.shape[1]
            w2 = x_target.shape[2]
            if self.block_dim == 1:
                xz = torch.cat([x_target, z_sample.repeat(1, w1, 1)], dim=-1)
            else:
                z_sample = z_sample.unsqueeze(1)
                xz = torch.cat([x_target, z_sample.repeat(1, w1, w2, 1)], dim=-1)

            y_pred, y_params = self.xz_to_y(xz)
            sigma = 0.1 + 0.9 * torch.sigmoid(y_pred[..., 1])
            p_y_pred = Normal(y_pred[..., 0], sigma)
            return p_y_pred, y_params, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample().unsqueeze(1)
            # Predict target points based on context
            w1 = x_target.shape[1]
            w2 = x_target.shape[2]
            if self.block_dim == 1:
                xz = torch.cat([x_target, z_sample.repeat(1, w1, 1)], dim=-1)
            else:
                z_sample = z_sample.unsqueeze(1)
                xz = torch.cat([x_target, z_sample.repeat(1, w1, w2, 1)], dim=-1)

            y_pred, y_params = self.xz_to_y(xz)
            sigma = 0.1 + 0.9 * torch.sigmoid(y_pred[..., 1])
            p_y_pred = Normal(y_pred[..., 0], sigma)
            return p_y_pred, y_params

    def count_params(self):
        return -1
