from typing import Iterator
from abc import ABC

import torch
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

import numpy as np
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO

from .lfm import LFM
from lafomo.utilities.torch import softplus, inv_softplus
from lafomo.configuration import VariationalConfiguration
from lafomo.datasets import LFMDataset


class VariationalLFM(LFM, ABC):
    """
    Variational inducing point approximation for Latent Force Models.

    Parameters
    ----------
    num_latents : int : the number of latent GPs (for example, the number of TFs)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    t_inducing : tensor of shape (..., T_u) : the inducing timepoints. Preceding dimensions are for multi-dimensional inputs
    """
    def __init__(self,
                 gp_model: ApproximateGP,
                 config: VariationalConfiguration,
                 dataset: LFMDataset,
                 dtype=torch.float64):
        super().__init__()
        self.gp_model = gp_model
        self.num_outputs = dataset.num_outputs
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        try:
            self.inducing_points = self.gp_model.get_inducing_points()
        except AttributeError:
            raise AttributeError('The GP model must define a function `get_inducing_points`.')

        num_training_points = self.inducing_points.numel()  # TODO num_data refers to the number of training datapoints

        self.loss_fn = VariationalELBO(self.likelihood, gp_model, num_training_points, combine_terms=False)
        self.config = config
        self.num_observed = dataset[0][0].shape[-1]
        self.dtype = dtype

        # if config.preprocessing_variance is not None:
        #     self.likelihood_variance = Parameter(torch.tensor(config.preprocessing_variance), requires_grad=False)
        # else:
        #     self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))

        if config.initial_conditions:
            self.initial_conditions = Parameter(torch.tensor(torch.zeros(self.num_outputs, 1)), requires_grad=True)

    def forward(self, x):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.gp_model.train(mode)
        self.likelihood.train(mode)

    def eval(self):
        self.gp_model.eval()
        self.likelihood.eval()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return [
            *self.gp_model.parameters(recurse),
            *super().parameters(recurse)
        ]

    def predict_m(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Calls self on input `t_predict`
        """
        return self.likelihood(self(t_predict.view(-1), **kwargs))

    def predict_f(self, t_predict) -> torch.distributions.MultivariateNormal:
        """
        Returns the latents
        """
        self.eval()
        with torch.no_grad():
            q_f = self.gp_model(t_predict)
        self.train()
        return q_f

    def save(self, filepath):
        torch.save(self.gp_model.state_dict(), 'gp-'+filepath+'.pt')
        torch.save(self.state_dict(), 'lfm-'+filepath+'.pt')

    @classmethod
    def load(cls,
             filepath,
             gp_cls,
             gp_args=[], gp_kwargs={},
             lfm_args=[], lfm_kwargs={}):
        gp_state_dict = torch.load('gp-'+filepath+'.pt')
        gp_model = gp_cls(*gp_args, **gp_kwargs)
        gp_model.load_state_dict(gp_state_dict)
        gp_model.double()

        lfm_state_dict = torch.load('lfm-'+filepath+'.pt')
        lfm = cls(gp_model, *lfm_args, **lfm_kwargs)
        lfm.load_state_dict(lfm_state_dict)
        return lfm
