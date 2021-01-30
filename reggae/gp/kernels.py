import torch
import gpytorch
from gpytorch.constraints import Positive, Interval

import numpy as np

PI = torch.tensor(np.pi, requires_grad=False)

class FirstGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_t, train_y, likelihood):
        super().__init__(train_t, train_y, likelihood)
        self.Y_var = Y_var
        self.num_genes = train_y.shape[0]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SIMKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SIMKernel(gpytorch.kernels.Kernel):
    """
    This kernel is the multi-output cross-kernel for linear response to single transcription factor.
    In other words, it constructs a JTxJT matrix where J is num genes and T is num timepoints.
    """

    is_stationary = True

    def __init__(self, num_genes, variance, **kwargs):
        super().__init__(**kwargs)
        self.num_genes = num_genes
        self.register_parameter(
            name='raw_lengthscale', parameter=torch.nn.Parameter(1.414 * torch.ones(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_decay', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, self.num_genes))
        )
        self.register_parameter(
            name='raw_sensitiviy', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, self.num_genes))
        )
        self.register_parameter(
            name='raw_scale', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )

        # register the constraints
        self.pos_contraint = Positive()
        self.decay_constraint = Interval(0.1, 1.5)
        self.sensitivity_constraint = Interval(0.1, 4)
        self.register_constraint("raw_lengthscale", self.pos_contraint)
        self.register_constraint("raw_decay", self.decay_constraint)
        self.register_constraint("raw_sensitiviy", self.sensitivity_constraint)
        self.register_constraint("raw_scale", self.pos_contraint)

        self.variance = torch.diag(variance)

    @property
    def lengthscale(self):
        return self.raw_lengthscale.constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale=self.raw_lengthscale.constraint.inverse_transform(value))
    @property
    def scale(self):
        return self.raw_lengthscale.constraint.transform(self.raw_lengthscale)

    @scale.setter
    def scale(self, value):
        self.initialize(raw_scale=self.raw_scale.constraint.inverse_transform(value))

    @property
    def decay(self):
        return self.raw_decay.constraint.transform(self.raw_decay)

    @decay.setter
    def decay(self, value):
        self.initialize(raw_decay=self.raw_decay.constraint.inverse_transform(value))

    @property
    def sensitivity(self):
        return self.raw_sensitivity.constraint.transform(self.raw_sensitivity)

    @sensitivity.setter
    def sensitivity(self, value):
        self.initialize(raw_sensitivity=self.raw_sensitivity.constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, **params):
        """
        This calculates Kxx (not cross-covariance)
        Parameters:
           x1 shape (num_genes*num_times)
        """
        # calculate the distance between inputs
        print(x1.shape, x2.shape)
        '''Computes Kxx'''
        self.block_size = int(x1.shape[0] / self.num_genes) # 7
        shape = [x1.shape[0], x2.shape[0]]
        K_xx = torch.zeros(shape)
        self.diff = self.covar_dist(x1, x2)

        for j in range(self.num_genes):
            for k in range(self.num_genes):
                kxx = self.k_xx(x1, j, k)
                K_xx[j * self.block_size:(j + 1) * self.block_size,
                     k * self.block_size:(k + 1) * self.block_size] = kxx

        # white = tf.linalg.diag(broadcast_tile(tf.reshape(self.noise_term, (1, -1)), 1, self.block_size)[0])
        jitter = 1e-6 * torch.eye(K_xx.shape[0])
        return K_xx + jitter + self.variance #+ white
        #
        #

    def k_xx(self, X, j, k, t_y=None):
        """k_xx(t, tprime)"""
        mult = self.sensitivity[j] * self.sensitivity[k] * self.lengthscale * 0.5 * torch.sqrt(PI)
        return self.scale ** 2 * mult * (self.h(X, k, j, t_y=t_y) + self.h(X, j, k, t_y=t_y, transpose=True))

    def h(self, X, k, j, t_y=None, primefirst=True):
        l = self.lengthscale
        #         print(l, self.D[k], self.D[j])
        t_x = X[:self.block_size].view(-1,)
        t_prime, t, t_dist = self.get_distance_matrix(primefirst=primefirst, t_x=t_x, t_y=t_y)
        multiplier = tfm.exp(self.gamma(k) ** 2) / (self.D[j] + self.D[k])
        first_erf_term = tfm.erf(t_dist / l - self.gamma(k)) + tfm.erf(t / l + self.gamma(k))
        second_erf_term = tfm.erf(t_prime / l - self.gamma(k)) + tfm.erf(self.gamma(k))
        return multiplier * (tf.multiply(tfm.exp(-self.D[k] * t_dist), first_erf_term) - \
                             tf.multiply(tfm.exp(-self.D[k] * t_prime - self.D[j] * t), second_erf_term))

    def K_xstarx(self, X, X2):
        """Computes Kx*,x
        Args:
          X:  x the blocked observation vector
          X2: x* the non-blocked prediction timepoint vector
        """
        self.block_size = X.shape[0]
        self.hori_block_size = int(X2.shape[0])
        self.vert_block_size = int(X.shape[0])
        shape = [X.shape[0] * self.num_genes, X2.shape[0] * self.num_genes]
        K_xx = tf.zeros(shape, dtype='float64')
        for j in range(self.num_genes):
            for k in range(self.num_genes):
                mask = np.ones(shape)
                other = np.zeros(shape)
                mask[j * self.vert_block_size:(j + 1) * self.vert_block_size,
                k * self.hori_block_size:(k + 1) * self.hori_block_size] = 0
                pad_top = j * self.vert_block_size
                pad_left = k * self.hori_block_size
                pad_right = 0 if k == self.num_genes - 1 else shape[1] - self.hori_block_size - pad_left
                pad_bottom = 0 if j == self.num_genes - 1 else shape[0] - self.vert_block_size - pad_top
                kxx = self.k_xx(X, j, k, t_y=X2)
                other = tf.pad(kxx, tf.constant([
                    [pad_top, pad_bottom],
                    [pad_left, pad_right]
                ]), 'CONSTANT')
                K_xx = K_xx * mask + other * (1 - mask)

        return K_xx

    def K_xf(self, X, X2):
        """Calculate K_xf: no need to use tf.* since this part is not optimised"""
        shape = [X.shape[0], X2.shape[0]]  # self.block_size]

        K_xf = tf.zeros(shape, dtype='float64')
        for j in range(self.num_genes):
            mask = np.ones(shape)
            other = np.zeros(shape)
            mask[j * self.block_size:(j + 1) * self.block_size] = 0
            pad_top = j * self.block_size
            pad_bottom = 0 if j == self.num_genes - 1 else shape[0] - self.block_size - pad_top
            kxf = self.k_xf(j, X, X2)
            other = tf.pad(kxf, tf.constant([[pad_top, pad_bottom], [0, 0]]), 'CONSTANT')

            K_xf = K_xf * mask + other * (1 - mask)
        return K_xf

    def k_xf(self, j, X, X2):
        t_prime, t_, t_dist = self.get_distance_matrix(t_x=tf.reshape(X[:self.block_size], (-1,)),
                                                       t_y=tf.reshape(X2, (-1,)))
        l = self.lengthscale
        erf_term = tfm.erf(t_dist / l - self.gamma(j)) + tfm.erf(t_ / l + self.gamma(j))
        return self.S[j] * l * 0.5 * tfm.sqrt(PI) * tfm.exp(self.gamma(j) ** 2) * tfm.exp(
            -self.D[j] * t_dist) * erf_term


    def gamma(self, k):
        return self.D[k] * self.lengthscale / 2


    def K_ff(self, X):
        """Returns the RBF kernel between latent TF"""
        _, _, t_dist = self.get_distance_matrix(t_x=tf.reshape(X, (-1,)))
        K_ff = self.kervar ** 2 * tfm.exp(-(t_dist ** 2) / (self.lengthscale ** 2))
        return (K_ff)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

