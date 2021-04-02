import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    NaturalVariationalDistribution,
    CholeskyVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
    TrilNaturalVariationalDistribution
)


class MultiOutputGP(ApproximateGP):
    def __init__(self,
                 inducing_points,
                 num_latents, use_scale=False,
                 use_ard=True,
                 initial_lengthscale=None,
                 lengthscale_constraint=None,
                 learn_inducing_locations=False,
                 natural=True,
                 use_tril=False):
        # The variational dist batch shape means we learn a different variational dist for each latent
        if natural:
            Distribution = TrilNaturalVariationalDistribution if use_tril else NaturalVariationalDistribution
            variational_distribution = Distribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        else:
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

        # Wrap the VariationalStrategy in a MultiTask to make output MultitaskMultivariateNormal
        # rather than a batch MVN
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations
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
        if use_scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                self.covar_module,
                batch_shape=torch.Size([num_latents])
            )
        if initial_lengthscale is not None:
            if use_scale:
                self.covar_module.base_kernel.lengthscale = initial_lengthscale
            else:
                self.covar_module.lengthscale = initial_lengthscale

    def get_inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    def forward(self, t):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(t)
        covar_x = self.covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
