import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class MultiOutputGP(ApproximateGP):
    def __init__(self, inducing_points, num_latents, use_ard=True, initial_lengthscale=None, lengthscale_constraint=None):
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ), num_tasks=num_latents
        )

        super().__init__(variational_strategy)

        # Modules should be marked as batch so different set of hyperparameters are learnt
        ard_dims = None
        if use_ard:
            ard_dims = inducing_points.shape[-1]
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(
            batch_shape=torch.Size([num_latents]),
            ard_num_dims=ard_dims,
            lengthscale_constraint=lengthscale_constraint
        )
        if initial_lengthscale is not None:
            self.covar_module.lengthscale = initial_lengthscale

    def get_inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    def forward(self, t):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(t)
        covar_x = self.covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
