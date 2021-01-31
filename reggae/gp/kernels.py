import torch
import gpytorch
from gpytorch.constraints import Positive, Interval

import numpy as np

PI = torch.tensor(np.pi, requires_grad=False)

from matplotlib import pyplot as plt


class SIMMean(gpytorch.means.Mean):

    def __init__(self, covar_module, num_genes, initial_basal):
        super().__init__()
        self.covar_module = covar_module
        self.pos_contraint = Positive()
        self.covar_module = covar_module
        self.num_genes = num_genes

        self.register_parameter(
            name='raw_basal', parameter=torch.nn.Parameter(
                self.pos_contraint.inverse_transform(0.5 * torch.ones(self.num_genes)))
        )
        self.register_constraint("raw_basal", self.pos_contraint)

        self.basal = initial_basal

    @property
    def basal(self):
        return self.pos_contraint.transform(self.raw_basal)

    @basal.setter
    def basal(self, value):
        self.initialize(raw_basal=self.pos_contraint.inverse_transform(value))

    def forward(self, x):
        block_size = int(x.shape[0]/self.num_genes)
        m = (self.basal / self.covar_module.decay).view(-1, 1)
        m = m.repeat(1, block_size).view(-1)
        return m


class SIMKernel(gpytorch.kernels.Kernel):
    """
    This kernel is the multi-output cross-kernel for linear response to single transcription factor.
    In other words, it constructs a JTxJT matrix where J is num genes and T is num timepoints.
    """

    is_stationary = True

    def __init__(self, num_genes, variance, **kwargs):
        super().__init__(**kwargs)
        self.num_genes = num_genes
        self.pos_contraint = Positive()
        self.lengthscale_constraint = Interval(0.5, 2.5)

        self.register_parameter(
            name='raw_lengthscale', parameter=torch.nn.Parameter(
                self.lengthscale_constraint.inverse_transform(1.414 * torch.ones(1, 1)))
        )
        self.register_parameter(
            name='raw_decay', parameter=torch.nn.Parameter(
                self.pos_contraint.inverse_transform(0.9 * torch.ones(self.num_genes)))
        )
        self.register_parameter(
            name='raw_sensitivity', parameter=torch.nn.Parameter(
                self.pos_contraint.inverse_transform(1 * torch.ones(self.num_genes)))
        )
        self.register_parameter(
            name='raw_scale', parameter=torch.nn.Parameter(0.5 * torch.ones(1, 1))
        )
        self.register_parameter(
            name='raw_noise', parameter=torch.nn.Parameter(8 * torch.ones(self.num_genes))
        )

        # register the constraints
        # self.decay_constraint = Interval(0.1, 1.5)
        # self.sensitivity_constraint = Interval(0.1, 4)
        self.register_constraint("raw_lengthscale", self.lengthscale_constraint)
        self.register_constraint("raw_decay", self.pos_contraint)
        self.register_constraint("raw_sensitivity", self.pos_contraint)
        self.register_constraint("raw_scale", self.pos_contraint)
        self.register_constraint("raw_noise", self.pos_contraint)

        self.variance = torch.diag(variance)

    @property
    def lengthscale(self):
        return self.lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale=self.lengthscale_constraint.inverse_transform(value))

    @property
    def scale(self):
        return self.pos_contraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self.initialize(raw_scale=self.pos_contraint.inverse_transform(value))

    @property
    def decay(self):
        return self.pos_contraint.transform(self.raw_decay)

    @decay.setter
    def decay(self, value):
        self.initialize(raw_decay=self.pos_contraint.inverse_transform(value))

    @property
    def sensitivity(self):
        return self.pos_contraint.transform(self.raw_sensitivity)

    @sensitivity.setter
    def sensitivity(self, value):
        self.initialize(raw_sensitivity=self.pos_contraint.inverse_transform(value))

    @property
    def noise(self):
        return self.pos_contraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self.initialize(raw_noise=self.pos_contraint.inverse_transform(value))


    def plot_cov(self, x1, x2):
        Kxx = self(x1, x2)
        plt.figure()
        plt.imshow(Kxx.detach().evaluate().detach())
        plt.colorbar()
        return Kxx

    def forward(self, x1, x2, **params):
        """
        This calculates Kxx (not cross-covariance)
        Parameters:
           x1 shape (num_genes*num_times)
        """
        # calculate the distance between inputs
        '''Computes Kxx'''
        self.block_size = int(x1.shape[0] / self.num_genes)  # 7
        shape = [x1.shape[0], x2.shape[0]]
        K_xx = torch.zeros(shape)
        self.diff = self.covar_dist(x1, x2)

        for j in range(self.num_genes):
            for k in range(self.num_genes):
                kxx = self.k_xx(j, k)
                K_xx[j * self.block_size:(j + 1) * self.block_size,
                k * self.block_size:(k + 1) * self.block_size] = kxx

        # white = tf.linalg.diag(broadcast_tile(tf.reshape(self.noise_term, (1, -1)), 1, self.block_size)[0])
        noise = self.noise.view(-1, 1).repeat(1, self.block_size).view(-1)
        noise = torch.diag(noise)

        # jitter = 1e-1 * torch.eye(self.block_size).repeat(self.num_genes, self.num_genes)
        jitter = 1e-1 * torch.eye(K_xx.shape[0])
        # plt.figure()
        # plt.imshow((self.variance+noise+jitter).detach())
        # plt.figure()
        return K_xx + jitter + self.variance + noise

    def k_xx(self, j, k):
        """k_xx(t, tprime)"""
        mult = self.sensitivity[j] * self.sensitivity[k] * self.lengthscale * 0.5 * torch.sqrt(PI)
        return self.scale ** 2 * mult * (self.h(k, j) + self.h(j, k, transpose=True))

    def h(self, k, j, transpose=False):
        l = self.lengthscale
        #         print(l, self.D[k], self.D[j])
        t = self.diff[0, :self.block_size]
        t_prime = t.repeat(self.block_size, 1)
        t = t.view(-1, 1).repeat(1, self.block_size)
        t_dist = self.diff[:self.block_size, :self.block_size]
        if transpose:
            t, t_prime = t_prime, t
            t_dist = torch.transpose(t_dist, 0, 1)

        multiplier = torch.exp(self.gamma(k) ** 2) / (self.decay[j] + self.decay[k])  # (1, 1)
        first_erf_term = torch.erf(t_dist / l - self.gamma(k)) + torch.erf(t / l + self.gamma(k))  # (T,T)
        second_erf_term = torch.erf(t_prime / l - self.gamma(k)) + torch.erf(self.gamma(k))
        return multiplier * (torch.multiply(torch.exp(-self.decay[k] * t_dist), first_erf_term) - \
                             torch.multiply(torch.exp(-self.decay[k] * t_prime - self.decay[j] * t), second_erf_term))

    def gamma(self, k):
        return self.decay[k] * self.lengthscale / 2

    def K_xstarx(self, x1, x2):
        """Computes Kx*,x
        Args:
          x1:  x the blocked observation vector
          x2: x* the non-blocked prediction timepoint vector
        """
        self.block_size = x1.shape[0]
        self.hori_block_size = int(x2.shape[0])
        self.vert_block_size = int(x1.shape[0])
        shape = [x1.shape[0] * self.num_genes, x2.shape[0] * self.num_genes]
        K_xx = torch.zeros(shape, dtype=torch.float32)
        for j in range(self.num_genes):
            for k in range(self.num_genes):
                kxx = self.k_xx(j, k, t_y=x2)
                K_xx[j * self.vert_block_size:(j + 1) * self.vert_block_size,
                     k * self.hori_block_size:(k + 1) * self.hori_block_size] = kxx

        return K_xx

    def K_xf(self, x, f):
        """
        K_xf
        Cross-covariance. Not optimised (not in marginal likelihood).
        Parameters:
            x: tensor (JT, JT)
            f: tensor (T*)
        """
        shape = [x.shape[0], f.shape[0]]
        K_xf = torch.zeros(shape, dtype=torch.float32)
        self.diff = self.covar_dist(x, f)[:self.block_size, :]
        print(self.block_size)
        for j in range(self.num_genes):
            kxf = self.k_xf(j, x, f.view(-1))
            print('kxf', kxf.shape)
            K_xf[j * self.block_size:(j + 1) * self.block_size] = kxf

        return K_xf

    def k_xf(self, j, x, f):
        l = self.lengthscale
        t = self.diff[0, :] #get first row
        t = t.repeat(self.block_size, 1) # may need to alter
        print(t.shape)
        erf_term = torch.erf(self.diff / l - self.gamma(j)) + torch.erf(t / l + self.gamma(j))
        return self.sensitivity[j] * l * 0.5 * torch.sqrt(PI) * torch.exp(self.gamma(j) ** 2) * torch.exp(
            -self.decay[j] * self.diff) * erf_term

    def K_ff(self, X):
        """Returns the RBF kernel between latent TF"""
        _, _, t_dist = self.get_distance_matrix(t_x=tf.reshape(X, (-1,)))
        K_ff = self.kervar ** 2 * tfm.exp(-(t_dist ** 2) / (self.lengthscale ** 2))
        return (K_ff)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))
