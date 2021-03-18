import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from lafomo.utilities.torch import softplus, inv_softplus


class RBF(Module):
    def __init__(self, num_outputs, scale=True, dtype=torch.float32):
        super(RBF, self).__init__()
        self.raw_lengthscale = Parameter(inv_softplus(1 * torch.ones((num_outputs), dtype=dtype)))
        self.raw_scale = Parameter(inv_softplus(torch.ones((num_outputs), dtype=dtype)), requires_grad=scale)
        self.num_outputs = num_outputs

    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = inv_softplus(value)

    @property
    def scale(self):
        return softplus(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self.raw_scale = inv_softplus(value)

    def forward(self, x: torch.Tensor, x2: torch.Tensor=None):
        """
        Radial basis function kernel.
        Parameters:
            x: tensor
            x2: if None, then x2 becomes x
        Returns:
             K of shape (I, |x|, |x2|)
        """
        add_jitter = x2 is None
        if x2 is None:
            x2 = x
        x = x.view(-1)
        x2 = x2.view(-1)
        sq_dist = torch.square(x.view(-1, 1)-x2)
        sq_dist = sq_dist.repeat(self.num_outputs, 1, 1)
        sq_dist = torch.div(sq_dist, 2*self.lengthscale.view((-1, 1, 1)))
        K = self.scale.view(-1, 1, 1) * torch.exp(-sq_dist)
        if add_jitter:
            jitter = 1e-5 * torch.eye(x.shape[0], dtype=K.dtype, device=K.device)
            K += jitter

        return K

    def __str__(self):
        return '%.02f' % self.lengthscale[0].item()


class SpatioTemporalRBF(RBF):
    def __init__(self, num_outputs, initial_lengthscale=1.5, scale=True, dtype=torch.float32):
        super().__init__(num_outputs, scale, dtype)

        lengthscales = torch.stack([
            inv_softplus(initial_lengthscale * torch.ones((num_outputs), dtype=dtype)),
            inv_softplus(initial_lengthscale * torch.ones((num_outputs), dtype=dtype))
        ], dim=1)
        print(lengthscales.shape)

        self.raw_lengthscale = Parameter(lengthscales, requires_grad=True)

    def forward(self, x: torch.Tensor, x2: torch.Tensor=None):
        """
        Radial basis function kernel.
        Parameters:
            x: tensor (2, T)
            x2: tensor (2, T'). if None, then x2 becomes x
        Returns:
             K of shape (I, |x|, |x2|)
        """
        add_jitter = x2 is None
        if x2 is None:
            x2 = x
        # (num_outputs, 2, N, N')

        l = 2 * self.lengthscale.view((self.num_outputs, 2, 1, 1))
        sq_dist = torch.square(x.view(1, 2, -1, 1)-x2.view(1, 2, 1, -1))
        sq_dist = torch.div(sq_dist, l).sum(dim=1)
        sq_dist = sq_dist.repeat(self.num_outputs, 1, 1)
        K = self.scale.view(-1, 1, 1) * torch.exp(-sq_dist)
        # from matplotlib import pyplot as plt
        # plt.imshow(K[0].detach())
        if add_jitter:
            jitter = 1e-5 * torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
            K += jitter

        return K

    def __str__(self):
        return '%.02f %.02f' % (self.lengthscale[0][0].item(), self.lengthscale[0][1].item())
