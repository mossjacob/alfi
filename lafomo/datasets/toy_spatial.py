import torch
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import wofz

from lafomo.datasets import LFMDataset
from lafomo.utilities.data import generate_neural_dataset

PI = torch.tensor(np.pi, requires_grad=False)


class ReactionDiffusion(LFMDataset):
    """
    Reaction diffusion equations with different hyperparameter settings.
    Generated using the `ReactionDiffusionGenerator`
    'l1', 'l2', 'decay', 'diffusion'
    """
    def __init__(self, data_dir='../data/', max_n=2000, nn_format=True, ntest=50):
        data = torch.load(Path(data_dir) / 'toydata.pt')
        orig_data = data['orig_data']
        x_observed = data['x_observed']
        self.num_data = min(orig_data.shape[0], max_n)
        self.num_outputs = 1
        self.num_discretised = 40
        self.orig_data = torch.cat([
            x_observed.unsqueeze(0).repeat(self.num_data, 1, 1),
            orig_data[:self.num_data]
        ], dim=1)
        params = data['params'][:self.num_data]

        if nn_format:
            train, test = generate_neural_dataset(self.orig_data, params, self.num_data - ntest, ntest)
            self.data = train
            self.train_data = train
            self.test_data = test
        else:
            self.data = [(x_observed, orig_data[i], params[i])
                         for i in range(self.num_data)]

        self.gene_names = np.array(['toy'])


class HomogeneousReactionDiffusion(LFMDataset):
    """
    Homogeneous meaning the decay, sensitivity, diffusion, and lengthscale hyperparameters are all fixed at:
    S=1, l1=0.3, l2=0.3, λ=0.1, D=0.01
    This is the toy dataset from López-Lopera et al. (2019)
    https://arxiv.org/abs/1808.10026
    Data download: https://github.com/anfelopera/PhysicallyGPDrosophila
    nn_format is compatible with neural operator models.
    """
    def __init__(self, data_dir='../data/', one_fixed_sample=True, highres=False, nn_format=None, ntest=50, sub=1):
        if one_fixed_sample:
            data = pd.read_csv(Path(data_dir) / 'demToy1GPmRNA.csv')
            nn_format = False
        else:
            if highres:
                data = pd.read_csv(Path(data_dir) / 'toy_GPmRNA_N50highres.csv')
            else:
                data = pd.read_csv(Path(data_dir) / 'toy_GPmRNA_N1050.csv')
            nn_format = True if nn_format is None else nn_format
        num_per_data = np.unique(data.values[:, 0]).shape[0] * np.unique(data.values[:, 1]).shape[0]
        self.num_data = data.values.shape[0] // num_per_data

        self.orig_data = torch.tensor(data.values).reshape(self.num_data, num_per_data, 4).permute(0, 2, 1)
        self.num_outputs = 1

        x_observed = torch.tensor(data.values[:, 0:2]).permute(1, 0)
        num_data = x_observed.shape[1] // num_per_data
        data = torch.tensor(data.values[:, 3]).unsqueeze(0)
        self.num_discretised = 40
        self.gene_names = np.array(['toy'])

        if nn_format:
            params = torch.tensor([0.3, 0.3, 0.1, 0.01]).unsqueeze(0).repeat(self.orig_data.shape[0], 1)
            train, test = generate_neural_dataset(self.orig_data, params, self.num_data - ntest, ntest, sub=sub)
            self.data = train
            self.train_data = train
            self.test_data = test
        else:
            self.data = [
                (x_observed[:, num_per_data*i:num_per_data*(i+1)], data[:, num_per_data*i:num_per_data*(i+1)])
                for i in range(num_data)
            ]


class ReactionDiffusionGenerator:
    """
    Reaction Diffusion LFM generator (transcribed from R to Python by Jacob Moss).
    The original author for the R code for the reaction diffusion equation is López-Lopera et al. (2019)
    https://github.com/anfelopera/PhysicallyGPDrosophila
    """

    def __init__(self, lengthscale=None, decay=0.1, diffusion=0.01):
        super().__init__()
        l = [0.3, 0.3] if lengthscale is None else lengthscale
        self.lengthscale = torch.tensor(l, dtype=torch.float64)
        self.sensitivity = torch.tensor(1., dtype=torch.float64)
        self.decay = torch.tensor(decay, dtype=torch.float64)
        self.diffusion = torch.tensor(diffusion, dtype=torch.float64)

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
        return torch.erf(arg1).t() - torch.erf(arg2)

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
        cache = dict()
        for i in range(1, nterms + 1):
            for j in range(1, nterms + 1):
                if ((i + j) % 2) == 0:
                    # computing kernel
                    key = f'{j},{i}'
                    if key in cache.keys():
                        kernt = cache[key][0].t()
                        kernx = cache[key][1].t()
                    else:
                        kernt = self.simXsimKernCompute(tx1[:, 0], tx2[:, 0], i, j, l)
                        kernx = self.sheatXsheatKernCompute(tx1[:, 1], tx2[:, 1], i, j, l)
                        key = f'{i},{j}'
                        cache[key] = [kernt, kernx]

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
        tx1 = tx1.type(torch.float64)
        tx2 = tx2.type(torch.float64)
        Kuu = Kyy = Kuy = Kyu = torch.zeros(10)
        # Kuu = self.prior(tx1[:, :2], tx2[:, :2])
        Kuu = self.kuu(tx1, tx2)
        Kyy = self.kyy(tx1, tx2)
        Kuy = self.kuy(tx1, tx2)
        Kyu = Kuy.t()  # self.kyu(tx1, tx2)
        return Kuu, Kyy, Kyu, Kuy

    def save_dataset(self, data_dir='../data'):
        """
        data_dir: the directory where the toy data and intermediate data lies. Will also be saved here.
        """
        temp = pd.read_csv(Path(data_dir) / 'demToy1GPmRNA.csv').values
        toydata = torch.load(Path(data_dir) / 'intermediate_toydata.pt')
        params_list = list()
        orig_data = list()
        num_samples = toydata[0]['samples'].shape[0]
        x_observed = torch.tensor(temp[:, 0:2]).permute(1, 0)

        for i in range(len(toydata)):
            params = torch.tensor([toydata[i][key] for key in ['l1', 'l2', 'decay', 'diffusion']])
            samples = toydata[i]['samples']
            for sample in range(num_samples):
                lf = samples[sample, 1681:]
                out = samples[sample, :1681]
                lf_out = torch.stack([lf, out], dim=0)
                orig_data.append(lf_out)
                params_list.append(params)
        params = torch.stack(params_list)
        orig_data = torch.stack(orig_data)
        shuffle = torch.randperm(orig_data.size()[0])
        orig_data = orig_data[shuffle]
        params = params[shuffle]
        torch.save({'x_observed': x_observed, 'orig_data': orig_data, 'params': params}, Path(data_dir) / 'toydata.pt')
