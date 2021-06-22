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
                 mean_module,
                 covar_module,
                 inducing_points,
                 num_latents,
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

        self.mean_module = mean_module
        self.covar_module = covar_module

    def get_inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    def forward(self, t):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(t)
        covar_x = self.covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def generate_multioutput_gp(num_latents, inducing_points,
                            kernel_class=gpytorch.kernels.RBFKernel,
                            kernel_kwargs=None,
                            ard_dims=None,
                            use_scale=False,
                            initial_lengthscale=None,
                            lengthscale_constraint=None,
                            zero_mean=True,
                            gp_kwargs=None):
    # Modules should be marked as batch so different set of hyperparameters are learnt
    if gp_kwargs is None:
        gp_kwargs = {}
    if kernel_kwargs is None:
        kernel_kwargs = {}
    if zero_mean:
        mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
    else:
        mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
    covar_module = kernel_class(
        batch_shape=torch.Size([num_latents]),
        ard_num_dims=ard_dims,
        lengthscale_constraint=lengthscale_constraint,
        **kernel_kwargs
    )
    if use_scale:
        covar_module = gpytorch.kernels.ScaleKernel(
            covar_module,
            batch_shape=torch.Size([num_latents])
        )
    if initial_lengthscale is not None:
        if use_scale:
            covar_module.base_kernel.lengthscale = initial_lengthscale
        else:
            covar_module.lengthscale = initial_lengthscale
    return MultiOutputGP(mean_module, covar_module, inducing_points, num_latents, **gp_kwargs)
