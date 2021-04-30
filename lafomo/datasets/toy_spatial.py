import torch
import numpy as np

from lafomo.datasets import LFMDataset
from scipy.special import wofz

PI = torch.tensor(np.pi, requires_grad=False)


class ToyReactionDiffusion(LFMDataset):
    """

    """

    def __init__(self):
        super().__init__()
        self.lengthscale = [0.3, 0.3]
        self.sensitivity = torch.tensor(1.)
        self.decay = torch.tensor(0.1)
        self.diffusion = torch.tensor(0.01)

    def theta_x(self):
        return self.lengthscale[1]

    def theta_t(self):
        return self.lengthscale[0]

    def Hfun(self, beta_s, t1, t2):
        # t1, t2 = vectors with  (time coordinates)
        # precomputing some terms
        v = 0.5 * self.theta_t() * beta_s
        # diff_t = outer(t2, t1,'-')
        diff_t = t2.view(-1, 1) - t1.view(-1)

        arg1 = v + t1 / self.theta_t()
        arg2 = v - diff_t / self.theta_t()

        # computing the function H for the sim kernel
        H = torch.erf(arg1).t() - torch.erf(arg2)
        return H  # .real TODO

    def hfun(self, t1, t2, beta_s, beta_q):
        # t1, t2 = vectors with  (time coordinates)
        v = 0.5 * self.theta_t() * beta_s
        # print('v', v, self.theta_t(), beta_s)
        # diff_t = outer(t2, t1,'-')
        diff_t = t2.view(-1, 1) - t1.view(-1)
        t0 = 0. * t1
        # computing the function h for the sim kernel
        hpart1 = torch.exp(-beta_s * diff_t) * self.Hfun(beta_s, t1, t2)
        thing = (beta_s * t2).view(-1, 1) + (beta_q * t1).view(-1)
        # outer(beta_s*t2,beta_q*t1,'+')

        hpart2 = torch.exp(-thing) * self.Hfun(beta_s, t0, t2)
        # print(self.Hfun(beta_s,t0,t2).min().item())
        # print(self.Hfun(beta_s,t0,t2).max().item())

        # print('sub', torch.exp(v**2), (hpart1 - hpart2)/(beta_s + beta_q))
        h = torch.exp(v ** 2) * (hpart1 - hpart2) / (beta_s + beta_q)
        return h  # .real

    def simXsimKernCompute(self, t1, t2, n, m, l):
        # t1, t2 = vectors with  (time coordinates)
        omega_n2 = (n * np.pi / l) ** 2
        omega_m2 = (m * np.pi / l) ** 2

        beta_q = self.decay + self.diffusion * omega_n2
        beta_s = self.decay + self.diffusion * omega_m2
        # computing the sim kernel
        # a = self.hfun(t1, t2, beta_s, beta_q).t()
        # print(a.min())
        # print(a.max())
        kern = self.hfun(t1, t2, beta_s, beta_q).t() + \
               self.hfun(t2, t1, beta_q, beta_s)

        # print('hfun before1', self.hfun(t1, t2, beta_s, beta_q).t())
        # print('hfun before2', self.hfun(t2, t1, beta_q, beta_s))
        # print('kern hfun', kern)
        kern = 0.5 * np.sqrt(np.pi) * self.theta_t() * kern

        return kern

    def sheatXsheatKernCompute(self, x1, x2, n, m, l):
        # x1, x2 = vectors with  (space coordinates)
        theta_x = self.theta_x()
        omega_n = n * np.pi / l
        gamma_n = 1j * omega_n
        z1_n = 0.5 * theta_x * gamma_n
        z2_n = l / theta_x + z1_n
        wz1_n = wofz(1j * z1_n)
        wz2_n = wofz(1j * z2_n)
        Wox_n = wz1_n - torch.exp(-(l / theta_x) ** 2 - gamma_n * l) * wz2_n
        if n == m:
            omega_m = omega_n
            c = ((theta_x * n * np.pi) ** 2 + 2 * l ** 2) / (2 * np.pi * n * l ** 2)
            c = Wox_n.real - Wox_n.imag * c
            c = 0.5 * theta_x * np.sqrt(np.pi) * l * c + \
                0.5 * theta_x ** 2 * (torch.exp(-(l / theta_x) ** 2) * np.cos(n * np.pi) - 1)
        else:
            omega_m = m * np.pi / l
            gamma_m = 1j * omega_m
            z1_m = 0.5 * theta_x * gamma_m
            z2_m = l / theta_x + z1_m
            wz1_m = wofz(1j * z1_m)
            wz2_m = wofz(1j * z2_m)
            Wox_m = wz1_m - torch.exp(-(l / theta_x) ** 2 - gamma_m * l) * wz2_m
            c = n * Wox_m.imag - m * Wox_n.imag
            c = c * (theta_x * l / (np.sqrt(np.pi) * (m ** 2 - n ** 2)))

        # computing the sheat kernel
        # kern = outer(torch.sin(omega_n*x1),
        #              torch.sin(omega_m*x2),'*') * c
        # print(torch.sin(omega_n*x1).shape, c.shape)
        kern = torch.outer(torch.sin(omega_n * x1),
                           torch.sin(omega_m * x2))
        kern = kern * c
        return kern

    def sheatXrbfKernCompute(self, x1, x2, n, l):
        # x1, x2 = vectors with  (space coordinates)
        # precomputing some terms
        omega_n = n * np.pi / l
        gamma_n = 1j * omega_n
        z1_n = 0.5 * self.theta_x() * gamma_n + x2 / self.theta_x()
        z2_n = z1_n - l / self.theta_x()
        wz1_n = wofz(1j * z1_n)
        wz2_n = wofz(1j * z2_n)
        Wox_n = np.exp(-((x2 - l) / self.theta_x()) ** 2 + gamma_n * l) * wz2_n - \
                np.exp(-(x2 / self.theta_x()) ** 2) * wz1_n
        c = 0.5 * np.sqrt(np.pi) * self.theta_x() * Wox_n.imag

        # computing the sheatXrbf kernel
        # kern = outer(sin(omega_n*x1), c)
        kern = torch.outer(torch.sin(omega_n * x1), c)

        return kern

    def simXrbfKernCompute(self, t1, t2, n, l):
        # t1, t2 = vectors with  (time coordinates)
        # precomputing some terms
        omega_n2 = (n * np.pi / l) ** 2
        beta_q = self.decay + self.diffusion * omega_n2
        z1 = 0.5 * self.theta_t() * beta_q
        # diff_t = outer(t1,t2,'-')
        diff_t = t1.view(-1, 1) - t2.view(-1)
        # computing the simXrbf kernel
        hterm = self.Hfun(beta_q, t2, t1)

        kern = np.exp(z1 ** 2 - beta_q * diff_t) * hterm
        kern = 0.5 * np.sqrt(np.pi) * self.theta_t() * kern
        return kern

    def kyy(self, tx1, tx2):
        nterms = 10
        l = 1
        kern = 0
        for i in range(1, nterms + 1):
            print(i)
            for j in range(1, nterms + 1):
                if ((i + j) % 2) == 0:
                    # computing kernel
                    kernt = self.simXsimKernCompute(tx1[:, 0], tx2[:, 0], i, j, l)
                    kernx = self.sheatXsheatKernCompute(tx1[:, 1], tx2[:, 1], i, j, l)
                    # print('kernt', kernt)
                    # print(kernt.shape, kernx.shape)
                    kern = kern + kernt * kernx

        kern = ((2 * self.sensitivity / l) ** 2) * kern
        return kern

    def kuy(self, x1, x2):
        # x1, x2 = matrices with  (time coordinate, space coordinate)
        # par = (sigma2, theta_t, theta_x, decay (lambda), diffusion (D), sensitivity (S))
        nterms = 10
        l = 1
        kern = 0

        # computing kernel and its gradient
        for i in range(1, nterms + 1):
            # computing kernel
            kernt = self.simXrbfKernCompute(x2[:, 0], x1[:, 0], i, l)
            kernx = self.sheatXrbfKernCompute(x2[:, 1], x1[:, 1], i, l)
            kern = kern + kernt * kernx

        kern = (2 * self.sensitivity / l) * kern.t()

        return kern

    def kuu(self, tx1, tx2):
        t1 = tx1[:, 0]
        t2 = tx2[:, 0]
        x1 = tx1[:, 1]
        x2 = tx2[:, 1]
        theta_t = self.theta_t()
        theta_x = self.theta_x()
        diff_t = t1.view(-1, 1) / theta_t - t2.view(-1) / theta_t
        diff_x = x1.view(-1, 1) / theta_x - x2.view(-1) / theta_x
        dist_t = torch.square(diff_t)
        dist_x = torch.square(diff_x)
        # computing the SE kernel
        return torch.exp(-dist_t - dist_x)

    def joint(self, tx1, tx2):
        Kuu = Kyy = Kuy = Kyu = torch.zeros(10)
        # Kuu = self.prior(tx1[:, :2], tx2[:, :2])
        Kuu = self.kuu(tx1, tx2)
        Kyy = self.kyy(tx1, tx2)
        Kuy = self.kuy(tx1, tx2)
        Kyu = Kuy.t()  # self.kyu(tx1, tx2)
        return Kuu, Kyy, Kyu, Kuy
