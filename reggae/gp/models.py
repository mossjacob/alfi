import gpflow
import numpy as np

import tensorflow as tf
from gpflow.models import GPR

from reggae.gp.options import Options
from reggae.data_loaders import DataHolder
from reggae.gp import LinearResponseKernel, LinearResponseMeanFunction


class LinearResponseModel(GPR):
    r"""
    Analytical Linear Response
    The kernel is defined in terms of m
    .. math::
       :nowrap:
       \begin{align}
           \theta        & \sim p(\theta) \\
           f             & \sim \mathcal{GP}(m(x), k(x, x'; \theta)) \\
           f_i           & = f(x_i) \\
           y_i \,|\, f_i & \sim p(y_i|f_i)
       \end{align}


    """

    def __init__(self, data: DataHolder, options: Options, replicate=0):
        self.data = data
        self.num_genes = data.m_obs.shape[1]
        self.N_m = data.m_obs.shape[2]
        Y = data.m_obs[replicate]
        Y_var = data.σ2_m_pre[replicate]
        self.Y = Y.reshape((-1, 1))
        self.Y_var = Y_var.reshape(-1)
        X = np.arange(self.N_m, dtype='float64') * 2
        self.X = np.c_[[X for _ in range(self.num_genes)]].reshape(-1, 1)

        self.kernel = LinearResponseKernel(data, self.Y_var)
        self.mean_function = LinearResponseMeanFunction(data, self.kernel)
        super().__init__(
            data=(self.X, self.Y),
            kernel=self.kernel,
            mean_function=self.mean_function
        )

    def predict_m(self, pred_t, compute_var=True):
        K_xx = self.kernel.K(self.X, None)
        K_inv = tf.linalg.inv(K_xx)
        K_xstarx = tf.transpose(self.kernel.K_xstarx(self.X[:self.N_m], pred_t))
        K_xstarxK_inv = tf.matmul(K_xstarx, K_inv)
        KxstarxKinvY = tf.linalg.matvec(K_xstarxK_inv, tf.reshape(self.Y, -1))
        mu = tf.reshape(KxstarxKinvY, (self.num_genes, pred_t.shape[0]))
        if compute_var:
            K_xstarxstar = self.kernel.K_xstarx(pred_t, pred_t)
            var = K_xstarxstar - tf.matmul(K_xstarxK_inv, tf.transpose(K_xstarx))
            var = tf.reshape(tf.linalg.diag_part(var), (self.num_genes, pred_t.shape[0]))
            return mu, var
        return mu

    def predict_f(self, pred_t, **kwargs):
        Kxx = self.kernel.K(self.X, None)
        K_inv = tf.linalg.inv(Kxx)
        Kxf = self.kernel.K_xf(self.X, pred_t)
        KfxKxx = tf.matmul(tf.transpose(Kxf), K_inv)
        return tf.reshape(tf.matmul(KfxKxx, self.Y), -1)
