# pylint: disable=E1136
import gpflow
from gpflow.utilities import positive
from gpflow.kernels import IndependentLatent, Kernel, Combination
import numpy as np

import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc
import tensorflow_probability as tfp

from reggae.utilities import broadcast_tile, PI

'''
This kernel is the multi-output cross-kernel for linear response to single transcription factor.
In other words, it constructs a JTxJT matrix where J is num genes and T is num timepoints.
'''
class LinearResponseKernel(gpflow.kernels.Kernel):
    def __init__(self, data, Y_var):
        super().__init__(active_dims=[0])
        self.Y_var = Y_var
        self.num_genes = data.m_obs.shape[1]
        #         l_affine = tfb.AffineScalar(shift=tf.cast(1., tf.float64),
        #                             scale=tf.cast(4-1., tf.float64))
        #         l_sigmoid = tfb.Sigmoid()
        #         l_logistic = tfb.Chain([l_affine, l_sigmoid])

        self.lengthscale = gpflow.Parameter(1.414, transform=positive())

        D_affine = tfb.AffineScalar(shift=tf.cast(0.1, tf.float64),
                                    scale=tf.cast(1.5-0.1, tf.float64))
        D_sigmoid = tfb.Sigmoid()
        D_logistic = tfb.Chain([D_affine, D_sigmoid])
        S_affine = tfb.AffineScalar(shift=tf.cast(0.1, tf.float64),
                                    scale=tf.cast(4.-0.1, tf.float64))
        S_sigmoid = tfb.Sigmoid()
        S_logistic = tfb.Chain([S_affine, S_sigmoid])

        self.D = gpflow.Parameter(np.random.uniform(0.9, 1, self.num_genes), transform=positive(), dtype=tf.float64)
        #         self.D[3].trainable = False
        #         self.D[3].assign(0.8)
        self.S = gpflow.Parameter(np.random.uniform(1,1, self.num_genes), transform=positive(), dtype=tf.float64)
        #         self.S[3].trainable = False
        #         self.S[3].assign(1)
        self.kervar = gpflow.Parameter(np.float64(1), transform=positive())
        self.noise_term = gpflow.Parameter(0.1353*tf.ones(self.num_genes, dtype='float64'), transform=positive())
        
    def K_xstarx(self, X, X2):
        '''Computes Kx*,x
        Args:
          X:  x the blocked observation vector
          X2: x* the non-blocked prediction timepoint vector
        '''
        self.block_size = X.shape[0]
        self.hori_block_size = int(X2.shape[0])
        self.vert_block_size = int(X.shape[0])
        shape = [X.shape[0]*self.num_genes, X2.shape[0]*self.num_genes]
        K_xx = tf.zeros(shape, dtype='float64')
        for j in range(self.num_genes):
            for k in range(self.num_genes):
                mask = np.ones(shape)
                other = np.zeros(shape)
                mask[j*self.vert_block_size:(j+1)*self.vert_block_size, 
                     k*self.hori_block_size:(k+1)*self.hori_block_size] = 0
                pad_top = j*self.vert_block_size
                pad_left = k*self.hori_block_size
                pad_right = 0 if k == self.num_genes-1 else shape[1]-self.hori_block_size-pad_left
                pad_bottom = 0 if j == self.num_genes-1 else shape[0]-self.vert_block_size-pad_top
                kxx = self.k_xx(X, j, k, t_y=X2)
                other = tf.pad(kxx, tf.constant([
                    [pad_top,pad_bottom],
                    [pad_left,pad_right]
                ]), 'CONSTANT')
                K_xx = K_xx * mask + other * (1 - mask)

        return K_xx

    def K(self, X, X2=None):
        '''Computes Kxx'''
        self.block_size = int(X.shape[0]/self.num_genes)
        if X2 is None:
            shape = [X.shape[0],X.shape[0]]
            K_xx = tf.zeros(shape, dtype='float64')
            for j in range(self.num_genes):
                for k in range(self.num_genes):
                    mask = np.ones(shape)
                    other = np.zeros(shape)
                    mask[j*self.block_size:(j+1)*self.block_size, 
                         k*self.block_size:(k+1)*self.block_size] = 0

                    pad_top = j*self.block_size
                    pad_left = k*self.block_size
                    pad_right = 0 if k == self.num_genes-1 else shape[0]-self.block_size-pad_left
                    pad_bottom = 0 if j == self.num_genes-1 else shape[0]-self.block_size-pad_top
                    kxx = self.k_xx(X, j, k)
                    other = tf.pad(kxx,
                                      tf.constant([
                                          [pad_top,pad_bottom],
                                          [pad_left,pad_right]
                                   ]), 'CONSTANT'
                                  )
                    K_xx = K_xx * mask + other * (1 - mask)

    #         K_xx = self.k_xx(X, 0,0)        
            white = tf.linalg.diag(broadcast_tile(tf.reshape(self.noise_term, (1, -1)), 1, self.block_size)[0])
            return K_xx + tf.linalg.diag((1e-5*tf.ones(X.shape[0], dtype='float64'))+self.Y_var) + white
        else:
            print('X not none', X2)
            return self.K_xf(X, X2)

    def K_xf(self, X, X2):
        '''Calculate K_xf: no need to use tf.* since this part is not optimised'''
        shape = [X.shape[0],X2.shape[0]]#self.block_size]

        K_xf = tf.zeros(shape, dtype='float64')
        for j in range(self.num_genes):
            mask = np.ones(shape)
            other = np.zeros(shape)
            mask[j*self.block_size:(j+1)*self.block_size] = 0
            pad_top = j*self.block_size
            pad_bottom = 0 if j == self.num_genes-1 else shape[0]-self.block_size-pad_top
            kxf = self.k_xf(j, X, X2)
            other = tf.pad(kxf, tf.constant([[pad_top,pad_bottom],[0,0]]), 'CONSTANT')

            K_xf = K_xf * mask + other * (1 - mask)
        return K_xf
        
    def k_xf(self, j, X, X2):
        t_prime, t_, t_dist = self.get_distance_matrix(t_x=tf.reshape(X[:self.block_size], (-1,)), 
                                                       t_y=tf.reshape(X2, (-1, )))
        l = self.lengthscale
        erf_term = tfm.erf(t_dist/l - self.gamma(j)) + tfm.erf(t_/l + self.gamma(j))
        return self.S[j]*l*0.5*tfm.sqrt(PI)*tfm.exp(self.gamma(j)**2) *tfm.exp(-self.D[j]*t_dist)*erf_term 

    def _gamma(self):
        return self.D*self.lengthscale/2

    def h(self, X, k, j, t_y=None, primefirst=True):
        l = self.lengthscale
#         print(l, self.D[k], self.D[j])
        t_x = tf.reshape(X[:self.block_size], (-1,))
        t_prime, t, t_dist = self.get_distance_matrix(primefirst=primefirst, t_x=t_x, t_y=t_y)
        multiplier = tfm.exp(self.gamma(k)**2) / (self.D[j]+self.D[k])
        first_erf_term = tfm.erf(t_dist/l - self.gamma(k)) + tfm.erf(t/l + self.gamma(k))
        second_erf_term = tfm.erf(t_prime/l - self.gamma(k)) + tfm.erf(self.gamma(k))
        return multiplier * (tf.multiply(tfm.exp(-self.D[k]*t_dist) , first_erf_term) - \
                             tf.multiply(tfm.exp(-self.D[k]*t_prime-self.D[j]*t) , second_erf_term))
    
    def gamma(self, k):
        return self.D[k]*self.lengthscale/2

    def k_xx(self, X, j, k, t_y=None):
        '''k_xx(t, tprime)'''
        mult = self.S[j]*self.S[k]*self.lengthscale*0.5*tfm.sqrt(PI)
        return self.kervar**2*mult*(self.h(X, k, j, t_y=t_y) + self.h(X, j, k, t_y=t_y, primefirst=False))
    
    def h_(self, X, k, j, primefirst=True):
        Dj = tf.reshape(self.D, (1, -1))
        Dj = broadcast_tile(Dj, 1, 7)
        Dj = tf.tile(Dj, [35, 1])
        Dk = tf.reshape(self.D, (-1, 1)) 
        Dk = broadcast_tile(Dk, 7, 1)
        Dk = tf.tile(Dk, [1, 35])
        gk = tf.transpose(broadcast_tile(tf.reshape(self.gamma(), (-1, 1)), 7, 1))
        gk = tf.tile(gk, [35, 1])
        if not primefirst:
            Dk, Dj = Dj, Dk
            gk = tf.transpose(broadcast_tile(tf.reshape(self.gamma(), (1,-1)), 1, 7))
            gk = tf.tile(gk, [1, 35])

        l = self.lengthscale
        t_x = tf.reshape(X[:self.block_size], (-1,))
        t_prime, t, t_dist = self.get_distance_matrix(primefirst=primefirst, t_x=t_x)
        t_prime = tf.tile(t_prime, [5, 5])
        t = tf.tile(t, [5, 5])
        t_dist = tf.tile(t_dist, [5, 5])
        multiplier = tfm.exp(gk**2) / (Dj + Dk)
        first_erf_term = tfm.erf(t_dist/l - gk) + tfm.erf(t/l + gk)
        second_erf_term = tfm.erf(t_prime/l - gk) + tfm.erf(gk)
        return multiplier * (tf.multiply(tfm.exp(-Dk*t_dist) , first_erf_term) - \
                             tf.multiply(tfm.exp(-Dk*t_prime-Dj*t) , second_erf_term))


    def _k_xx(self, X, j, k):
        S_square = tf.matmul(tf.reshape(self.S, (-1, 1)), tf.reshape(self.S, (1, -1)))
        S_square = broadcast_tile(S_square, 7, 7)
        mult = S_square*self.lengthscale*0.5*tfm.sqrt(PI)
        return self.kervar**2*mult*(self.h(X, k, j) + self.h(X, j, k, primefirst=False))

    def get_distance_matrix(self, t_x, primefirst=True, t_y=None):
        if t_y is None:
            t_y = t_x
        t_1 = tf.transpose(tf.reshape(tf.tile(t_x, [t_y.shape[0]]), [ t_y.shape[0], t_x.shape[0]]))
        t_2 = tf.reshape(tf.tile(t_y, [t_x.shape[0]]), [ t_x.shape[0], t_y.shape[0]])
        if primefirst:
            return t_1, t_2, t_1-t_2
        return t_2, t_1, t_2-t_1
    
    def K_ff(self, X):
        print('k_diag')

        """I've used the fact that we call this method for K_ff when finding the covariance as a hack so
        I know if I should return K_ff or K_xx. In this case we're returning K_ff!!
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        _,_,t_dist = self.get_distance_matrix(t_x=tf.reshape(X, (-1,)))
        K_ff = self.kervar **2 * tfm.exp(-(t_dist**2)/(self.lengthscale**2))
        return (K_ff)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))


class LinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    def __init__(self, kernels, W, name=None):
        Combination.__init__(self, kernels=kernels, name=name)
        self.W = gpflow.Parameter(W)  # [P, L]

    @property
    def num_latent_gps(self):
        return self.W.shape[-1]  # L

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def Kgg(self, X, X2):
        return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [L, N, N2]

    def K(self, X, X2=None, full_output_cov=True):
        Kxx = self.Kgg(X, X2)  # [L, N, N2]
        W = broadcast_tile(self.W, 7, 1)
        print(W.shape)
        tf.print(W.shape)
        KxxW = Kxx[None, :, :, :] * W[:, :, None, None]  # [P, L, N, N2]
        if full_output_cov:
            WKxxW = tf.tensordot(W, KxxW, [[1], [1]])  # [P, P, N, N2]
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
        else:
            return tf.reduce_sum(W[:, :, None, None] * KxxW, [1])  # [P, N, N2]


    def K_diag(self, X, full_output_cov=True):
        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, L]
        W = broadcast_tile(self.W, 7, 1)
        # print(self.W)
        if full_output_cov:
            Wt = tf.transpose(W)  # [L, P]
            return tf.reduce_sum(
                K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :], axis=1
            )  # [N, P, P]
        else:
            return tf.linalg.matmul(
                K, W ** 2.0, transpose_b=True
            )  # [N, L]  *  [L, P]  ->  [N, P]
