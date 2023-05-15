import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import MultivariateNormalPrior
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class DeepKernelLFM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, train_f, likelihood, likelihood_f, x_operator, f_operator, num_functions=2, embedding_scale_bounds=None, embedder=None, kernel='rbf'):
        super(DeepKernelLFM, self).__init__(train_x, train_y, likelihood)
        self.train_y = train_y
        self.train_f = train_f
        self.likelihood_f = likelihood_f
        self.deepkernel = x_operator
        self.f_deepkernel = f_operator
        self.embedder = embedder
        self.mean_module = gpytorch.means.ConstantMean().type(torch.float64)
        self.mean_module_f = gpytorch.means.ConstantMean().type(torch.float64)
        # self.mean_1 = torch.nn.Parameter(torch.zeros(1))
        # self.mean_2 = torch.nn.Parameter(torch.zeros(1))
        if kernel == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=9).type(torch.float64)
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(ard_num_dims=9).type(torch.float64)
            )

        embedding_scale_bounds = (0., 1.) if embedding_scale_bounds is None else embedding_scale_bounds
        b = 1.
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-b, b)
        self.scale_to_bounds_em = gpytorch.utils.grid.ScaleToBounds(*embedding_scale_bounds)
        self.num_functions = num_functions
        self.hn = None  # (n_task, n_functions, embedding_size)
        self.debug = False

    def cat_embedding(self, t, is_blocked=True):
        """
        Concatenates the embedding onto t
        Args:
            t: (num_task, T, 1) or (num_task, T * n_functions, 1) if is_blocked
            is_blocked: boolean indicating whether t is a blocked timepoint vector
        """
        # Reshape inputs (T) -> (num_task, T * num_functions, 1)
        # inputs = self.scale_to_bounds(inputs)
        n_task = t.shape[0]
        if self.debug and not self.training:
            plt.figure()
            plt.plot(t[0].squeeze())
            plt.title('inputs')
            print("GET_INPUTS called", t.shape)

        embedding_size = self.hn.shape[-1]
        # embedding = self.scale_to_bounds_em(self.hn)
        embedding = self.hn
        num_blocks = embedding.shape[1]
        num_points = t.shape[1]
        block_size = num_points // self.num_functions
        if is_blocked:
            embedding_shaped = torch.empty(n_task, num_points, embedding_size)
            for i in range(num_blocks):
                start_index = i * block_size
                end_index = (i+1) * block_size
                embedding_shaped[:, start_index:end_index] = embedding[:, i].unsqueeze(1)
            embedding = embedding_shaped
        else:
            embedding_shaped = torch.empty(n_task, num_points, embedding_size)
            embedding_shaped[:, :] = embedding.sum(dim=1).unsqueeze(1)
            # embedding_shaped[:, :] = embedding[:, 0].unsqueeze(1) + embedding[:, 1].unsqueeze(1)
            embedding = embedding_shaped
            # embedding = embedding.reshape(embedding.shape[0],  1, -1).repeat(1, num_points, 1)
        # if diff > 0:
        #     embedding_shaped[:, num_train:num_train + diff // self.num_functions] = embedding[:, 0].unsqueeze(1)
        #     embedding_shaped[:, num_train + diff // self.num_functions:] = embedding[:, 1].unsqueeze(1)

        if self.debug and not self.training:
            print('embedding shape', embedding.shape)
            plt.figure()
            plt.plot(embedding.mean(-1).detach()[0])
            plt.title('emb')

        # embedding = embedding.reshape(self.num_functions, 1, embedding_size)  # (n_task, n_functions, 1, embedding_size)
        # embedding = embedding.repeat(1, num_train // self.num_functions, 1)  # -> (n_task, n_functions, T, embedding_size)
        # embedding = embedding.reshape(-1, embedding_size)  # -> (n_task, n_functions * T, embedding_size)
        # embedding = embedding.reshape(n_task, self.num_functions, 1, embedding_size)  # (n_task, n_functions, 1, embedding_size)
        # embedding = embedding.repeat(1, t_x.shape[1] // self.num_functions, 1)  # -> (n_task, n_functions, T, embedding_size)
        # embedding = embedding.reshape(n_task, -1, embedding_size)  # -> (n_task, n_functions * T, embedding_size)
        return torch.cat([t, embedding], dim=-1)  # (n_task, n_functions * T, embedding_size + 1

    def kxf(self, t_x, t_f):
        """
        First, convert the input times t_x and t_f into higher-dimensional objects,
        so we have a D-dimensional vector for each timepoint.
        Args:
            t_x: (num_task, T * n_functions, 1)
            t_f: (num_task, T', 1)
        """
        x_projected_a = self.project_x(t_x)
        # diff = x_projected_a[:, : x_projected_a.shape[1] // n_functions] - x_projected_a[:, x_projected_a.shape[1] // n_functions:]
        # print('xf projected diff', diff[diff < 1e-2])

        t_projected = self.project_f(t_f)
        return self.covar_module(x_projected_a, t_projected)

    def kff(self, ta, tb):
        t_projected_a = self.project_f(ta)
        # plt.figure()
        # plt.title('inputs to Kff, one task')
        # pca = PCA(n_components=2)
        # print('tproj', t_projected_a[0].shape)
        # x = pca.fit_transform(t_projected_a.detach()[0])
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # print(x.shape, ta.shape)
        # plt.scatter(x[:, 0], x[:, 1], c=ta[0].squeeze())
        if ta.shape[1] == tb.shape[1] and torch.all(ta == tb):
            return self.covar_module(t_projected_a)
        t_projected_b = self.project_f(tb)

        return self.covar_module(t_projected_a, t_projected_b)

    def kxx(self, xa, xb):
        x_projected_a = self.project_x(xa)
        # diff = x_projected_a[:, : x_projected_a.shape[1] // 2] - x_projected_a[:, x_projected_a.shape[1] // 2 : ]
        # print('projected diff', torch.sum(diff < 1e-4))
        # plt.figure()
        # plt.title('inputs to Kxx, one task')
        # pca = PCA(n_components=2)
        # print('xproj', xa.shape)
        # x = pca.fit_transform(x_projected_a.detach()[0])
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.scatter(x[:, 0], x[:, 1], c=xa[0].squeeze())

        if xa.shape[1] == xb.shape[1] and torch.all(xa == xb):
            return self.covar_module(x_projected_a)

        x_projected_b = self.project_x(xb)
        return self.covar_module(x_projected_a, x_projected_b)

    def scale(self, x):
        return x
        # return self.scale_to_bounds(x)

    def project_x(self, t):
        t_scaled = t / self.train_inputs[0].max()
        if type(self.deepkernel) is tuple:
            raise NotImplementedError('deepkernel should not be a tuple')
            # return torch.cat([
            #     torch.cat([
            #         self.scale(self.deepkernel[0](t_scaled[:, :t.shape[1] // self.num_functions])),
            #         t_scaled[:, :t.shape[1] // self.num_functions]
            #     ], dim=2),
            #     torch.cat([
            #         self.scale(self.deepkernel[1](t_scaled[:, t.shape[1] // self.num_functions:])),
            #         t_scaled[:, t.shape[1] // self.num_functions:]
            #     ], dim=2),
            # ], dim=1)
        else:
            t_emb = t_scaled

            if self.embedder is not None:
                t_emb = self.cat_embedding(t_scaled)
            # return torch.cat([self.scale_to_bounds(self.deepkernel(t_emb)), t_scaled], dim=2)
            # plt.figure()
            # print(t_emb[0].shape)
            # print(t_emb[0].mean(-1).view(5, -1))
            # plt.plot(t_emb[0].mean(-1).detach().view(5, -1).t(), linewidth=0.5)
            # plt.figure()
            
            emb = self.deepkernel(t_emb, use_output_head=True)
            # print('before', emb.shape, emb[1, :5])
            emb = self.scale(emb)
            # print('after', emb.shape, emb[1, :5])
            emb = torch.cat([emb, t_scaled], dim=-1)
            return emb

    def project_f(self, t):
        t_scaled = t / self.train_inputs[0].max()
        t_emb = t_scaled
        if self.embedder is not None:
            # t_emb = self.cat_embedding(t_scaled, is_blocked=False)
            embedding_size = self.hn.shape[-1]
            t_emb = torch.cat([t_scaled, torch.zeros(t_scaled.shape[0], t_scaled.shape[1], embedding_size)], dim=-1)
        # return torch.cat([self.scale_to_bounds(self.f_operator(t_emb)), t_scaled], dim=2)

        emb = self.deepkernel(t_emb, use_output_head=False)
        # print('f emb before', emb[1, :5])

        emb = self.scale(emb)
        emb = torch.cat([emb, t_scaled], dim=-1)

        # print('f emb after', emb[1, :5])
        return emb

    # def mean_module(self, xa):
    #     length = len(xa) // 2
    #     return torch.cat([self.mean_1.repeat(length), self.mean_2.repeat(length)])
    def compute_embedding(self, y_cond, x_cond=None):
        if self.embedder is not None:
            n_task = y_cond.shape[0]
            y_reshaped = y_cond.reshape(n_task, self.num_functions, -1).reshape(n_task * self.num_functions, -1, 1)  # (n_task, n_functions, T -> n_task * n_functions, T, 1)
            self.hn = self.embedder(y_reshaped, x_cond=x_cond).reshape(n_task, self.num_functions, -1)

    def __call__(self, *args, **kwargs):
        y_reshaped = args[0]
        self.compute_embedding(y_reshaped)

        return super().__call__(*args[1:], **kwargs)

    def latent_prior(self, x_pred):
        prior_mean = self.mean_module_f(x_pred)
        #torch.zeros(t_f.shape[1], dtype=torch.float64).unsqueeze(0).repeat(t_f.shape[0], 1)
        prior_cov = self.kff(x_pred, x_pred) + torch.eye(x_pred.shape[1], dtype=torch.float64) * 1e-7
        return MultivariateNormal(prior_mean, prior_cov)

    def output_prior(self, t_x):
        return MultivariateNormal(self.mean_module(t_x), self.kxx(t_x, t_x) + torch.eye(t_x.shape[1], dtype=torch.float64) * 1e-6)

    def get_conditioning_or_default(self, x_cond, y_cond):
        """
        Returns the conditioning data or the training conditioning if not exist.
        """
        if x_cond is None:
            x_cond = self.train_inputs[0]
        if y_cond is None:
            y_cond = self.train_y
        return x_cond, y_cond

    def predictive(self, y_cond, x_pred, x_cond_blocks=None):
        """
        Args:
            y_cond: the output conditioning data
            x_pred: the timepoints to be predicted
            x_cond_blocks: the time inputs that is conditioned (n_task, T*n_functions) if None, then train_inputs are used.
        """
        x_cond_blocks, y_cond = self.get_conditioning_or_default(x_cond_blocks, y_cond)
        kxx = self.kxx(x_cond_blocks, x_cond_blocks)
        kxstarxstar = self.kxx(x_pred, x_pred)
        kxstarx = self.kxx(x_pred, x_cond_blocks)
        kxstarx_kxx_x = self.mean_module(x_pred) + kxstarx.matmul(kxx.inv_matmul(y_cond.unsqueeze(-1))).squeeze(-1)

        kxx_kxstarx_kxx_kxxstar = kxstarxstar - kxstarx.matmul(kxx.inv_matmul(kxstarx.transpose(-2, -1).evaluate()))
        kxx_kxstarx_kxx_kxxstar += torch.eye(kxx_kxstarx_kxx_kxxstar.shape[-1]) * self.likelihood.noise
        return gpytorch.distributions.MultivariateNormal(kxstarx_kxx_x, kxx_kxstarx_kxx_kxxstar)

    def latent_predictive(self, x_pred, x_cond=None, y_cond=None):
        p = self.conditional_f_given_x(x_cond, x_pred=x_pred, y_cond=y_cond)
        return p.add_jitter(noise=self.likelihood_f.noise)

    def conditional_x_given_f(self, x_cond_blocks, f_cond=None):
        if f_cond is None:
            f_cond = self.train_f
        jitter = torch.eye(x_cond_blocks.shape[1] // self.num_functions) * 1e-7
        kxx = self.kxx(x_cond_blocks, x_cond_blocks)
        kff = self.kff(
            x_cond_blocks[:, :x_cond_blocks.shape[1] // self.num_functions],
            x_cond_blocks[:, :x_cond_blocks.shape[1] // self.num_functions]
        ) + jitter
        kxf = self.kxf(x_cond_blocks, x_cond_blocks[:, :x_cond_blocks.shape[1] // self.num_functions])
        mean_cond = self.mean_module(x_cond_blocks) + kxf.matmul(kff.inv_matmul(f_cond.unsqueeze(-1))).squeeze(-1)
        k_cond = kxx - kxf.matmul(kff.inv_matmul(kxf.transpose(-1, -2).evaluate()))

        return gpytorch.distributions.MultivariateNormal(mean_cond, k_cond)

    def conditional_f_given_x(self, x_cond_blocks, x_pred=None, y_cond=None):
        """
        Args:
            y_reshaped: outputs
            x_pred: the timepoints to be predicted
            y_cond: the outputs to be conditioned on
        """
        if x_pred is None:
            x_pred = x_cond_blocks[:, :x_cond_blocks.shape[1] // self.num_functions]
        x_cond_blocks, y_cond = self.get_conditioning_or_default(x_cond_blocks, y_cond)

        jitter = torch.eye(x_pred.shape[1]) * 1e-7
        kxx = self.kxx(x_cond_blocks, x_cond_blocks)
        kxf = self.kxf(x_cond_blocks, x_pred)  # (n_task, |x|*n_functions, |f|*n_functions)
        kff = self.kff(x_pred, x_pred) + jitter
        kfx_kxx_x = self.mean_module_f(x_pred) + kxf.transpose(-2, -1).matmul(kxx.inv_matmul(y_cond.unsqueeze(-1))).squeeze(-1)

        kff_kfx_kxx_kxf = kff - kxf.transpose(-2, -1).matmul(kxx.inv_matmul(kxf.evaluate()))

        return gpytorch.distributions.MultivariateNormal(kfx_kxx_x, kff_kfx_kxx_kxf)

    def forward(self, input):
        return self.conditional_f_given_x(input)
        # return self.conditional_x_given_f(input)
