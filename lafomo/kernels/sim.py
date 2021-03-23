import gpytorch
import torch
import numpy as np

from gpytorch.constraints import Positive, Interval

PI = torch.tensor(np.pi, requires_grad=False)


class SIMKernel(gpytorch.kernels.Kernel):
    """
    Simple Input Module Motif Kernel
    This kernel is the multi-output cross-kernel for linear response to single transcription factor.
    In other words, it constructs a JTxJT matrix where J is num genes and T is num timepoints.
    SIM: Single Input Module motif
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
            name='raw_noise', parameter=torch.nn.Parameter(4 * torch.ones(self.num_genes))
        )

        # register the constraints
        self.register_constraint("raw_lengthscale", self.lengthscale_constraint)
        self.register_constraint("raw_decay", self.pos_contraint)
        self.register_constraint("raw_sensitivity", self.pos_contraint)
        self.register_constraint("raw_noise", self.pos_contraint)

        self.variance = torch.diag(variance)

    @property
    def lengthscale(self):
        return self.lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale=self.lengthscale_constraint.inverse_transform(value))

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
        from matplotlib import pyplot as plt
        Kxx = self(x1, x2)
        plt.figure()
        plt.imshow(Kxx.detach().evaluate().detach())
        plt.colorbar()
        return Kxx

    def forward(self, t1, t2, **params):
        """
        This calculates Kxx (not cross-covariance)
        t1, t2 are blocked timepoint vectors
        Parameters:
           x1 shape (num_genes*num_times)
        """
        vert_block_size = int(t1.shape[0] / self.num_genes)
        hori_block_size = int(t2.shape[0] / self.num_genes)
        t1_block, t2_block = t1[:vert_block_size], t2[:hori_block_size]
        shape = [vert_block_size * self.num_genes, hori_block_size * self.num_genes]
        K_xx = torch.zeros(shape, dtype=torch.float64)
        for j in range(self.num_genes):
            for k in range(self.num_genes):
                kxx = self.k_xx(j, k, t1_block, t2_block)
                K_xx[j * vert_block_size:(j + 1) * vert_block_size,
                     k * hori_block_size:(k + 1) * hori_block_size] = kxx

        if hori_block_size == vert_block_size:
            # noise = self.noise.view(-1, 1).repeat(1, hori_block_size).view(-1)
            # noise = torch.diag(noise)
            jitter = 1e-4 * torch.eye(K_xx.shape[0], dtype=torch.float64)
            K_xx += jitter
            if K_xx.shape[0] == self.variance.shape[0]:
                K_xx += self.variance
        return K_xx

    def k_xx(self, j, k, t1_block, t2_block):
        """
        k_xx(t, t')
        t2 = t'
        Parameters:
            t1_block: tensor shape (T1,)
            t2_block: tensor shape (T2,)
        """
        t1_block = t1_block.view(-1, 1)
        t2_block = t2_block.view(1, -1)
        mult = self.sensitivity[j] * self.sensitivity[k] * self.lengthscale * 0.5 * torch.sqrt(PI)
        k_xx = mult * (self.h(k, j, t2_block, t1_block) + self.h(j, k, t1_block, t2_block))
        return k_xx

    def h(self, k, j, t2, t1):
        l = self.lengthscale
        t_dist = t2 - t1
        multiplier = torch.exp(self.gamma(k) ** 2) / (self.decay[j] + self.decay[k])  # (1, 1)
        first_erf_term = torch.erf(t_dist / l - self.gamma(k)) + torch.erf(t1 / l + self.gamma(k))  # (T,T)
        second_erf_term = torch.erf(t2 / l - self.gamma(k)) + torch.erf(self.gamma(k))
        return multiplier * (torch.multiply(torch.exp(-self.decay[k] * t_dist), first_erf_term) - \
                             torch.multiply(torch.exp(-self.decay[k] * t2 - self.decay[j] * t1), second_erf_term))

    def gamma(self, k):
        return self.decay[k] * self.lengthscale / 2

    def K_xf(self, x, f):
        """
        K_xf
        Cross-covariance. Not optimised (not in marginal likelihood).
        Parameters:
            x: tensor (JT, JT) the blocked observation vector
            f: tensor (T*) the non-blocked f prediction timepoint vector
        """
        shape = [x.shape[0], f.shape[0]]
        K_xf = torch.zeros(shape, dtype=torch.float32)
        self.block_size = int(x.shape[0] / self.num_genes)  # 7
        t1_block, t2_block = x[:self.block_size].view(-1, 1), f.view(1, -1)
        for j in range(self.num_genes):
            kxf = self.k_xf(j, t1_block, t2_block)
            K_xf[j * self.block_size:(j + 1) * self.block_size] = kxf

        return K_xf

    def k_xf(self, j, x, t_f):
        l = self.lengthscale
        t_dist = x - t_f
        erf_term = torch.erf(t_dist / l - self.gamma(j)) + torch.erf(t_f / l + self.gamma(j))
        return self.sensitivity[j] * l * 0.5 * torch.sqrt(PI) * torch.exp(self.gamma(j) ** 2) * torch.exp(
            -self.decay[j] * t_dist) * erf_term

    def K_ff(self, x1, x2):
        """Returns the RBF kernel between latent TF"""
        x1 = x1.view(-1)
        x2 = x2.view(-1)
        sq_dist = torch.square(x1.view(-1, 1)-x2)
        sq_dist = torch.div(sq_dist, 2*self.lengthscale.view((-1, 1)))
        K = torch.exp(-sq_dist)
        if K.shape[0] == K.shape[1]:
            jitter = 1e-3 * torch.eye(x1.shape[0])
            K += jitter
        return K
