from abc import ABC

import torch
from torch.nn.parameter import Parameter

from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from .lfm import LFM
from lafomo.configuration import VariationalConfiguration
from lafomo.mlls import MaskedVariationalELBO


class VariationalLFM(LFM, ABC):
    """
    Variational inducing point approximation for Latent Force Models.

    Parameters
    ----------
    num_outputs : int : the number of outputs (for example, the number of genes)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    """
    def __init__(self,
                 num_outputs: int,
                 gp_model: ApproximateGP,
                 config: VariationalConfiguration,
                 num_training_points=None,
                 dtype=torch.float64):
        super().__init__()
        self.gp_model = gp_model
        self.num_outputs = num_outputs
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        self.pretrain_mode = False
        try:
            self.inducing_points = self.gp_model.get_inducing_points()
        except AttributeError:
            raise AttributeError('The GP model must define a function `get_inducing_points`.')

        if num_training_points is None:
            num_training_points = self.inducing_points.numel()  # TODO num_data refers to the number of training datapoints

        self.loss_fn = MaskedVariationalELBO(self.likelihood, gp_model, num_training_points, combine_terms=False)
        self.config = config
        self.dtype = dtype

        # if config.preprocessing_variance is not None:
        #     self.likelihood_variance = Parameter(torch.tensor(config.preprocessing_variance), requires_grad=False)
        # else:
        #     self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))

        if config.initial_conditions:
            self.initial_conditions = Parameter(torch.tensor(torch.zeros(self.num_outputs, 1)), requires_grad=True)

    def nonvariational_parameters(self):
        variational_keys = dict(self.gp_model.named_variational_parameters()).keys()
        named_parameters = dict(self.named_parameters())
        return [named_parameters[key] for key in named_parameters.keys()
                if key[len('gp_model.'):] not in variational_keys]

    def variational_parameters(self):
        return self.gp_model.variational_parameters()

    def summarise_gp_hyp(self):
        # variational_keys = dict(self.gp_model.named_variational_parameters()).keys()
        # named_parameters = dict(self.named_parameters())
        #
        # return [named_parameters[key] for key in named_parameters.keys()
        #         if key[len('gp_model.'):] not in variational_keys]
        if self.gp_model.covar_module.lengthscale is not None:
            return self.gp_model.covar_module.lengthscale.detach().numpy()
        elif hasattr(self.gp_model.covar_module, 'base_kernel'):
            kernel = self.gp_model.covar_module.base_kernel
            if hasattr(kernel, 'kernels'):
                if hasattr(kernel.kernels[0], 'lengthscale'):
                    return kernel.kernels[0].lengthscale.detach().numpy()
            else:
                return self.gp_model.covar_module.base_kernel.lengthscale.detach().numpy()
        else:
            return ''

    def forward(self, x):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.gp_model.train(mode)
        self.likelihood.train(mode)

    def pretrain(self, mode=True):
        self.pretrain_mode = mode

    def eval(self):
        self.gp_model.eval()
        self.likelihood.eval()
        self.pretrain(False)

    def predict_m(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Calls self on input `t_predict`
        """
        return self(t_predict.view(-1), **kwargs)

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
        torch.save(self.gp_model.state_dict(), filepath+'gp.pt')
        torch.save(self.state_dict(), filepath+'lfm.pt')

    @classmethod
    def load(cls,
             filepath,
             gp_cls=None, gp_model=None,
             gp_args=[], gp_kwargs={},
             lfm_args=[], lfm_kwargs={}):
        assert not (gp_cls is None and (gp_model is None))
        gp_state_dict = torch.load(filepath+'gp.pt')
        if gp_cls is not None:
            gp_model = gp_cls(*gp_args, **gp_kwargs)
        gp_model.load_state_dict(gp_state_dict)
        gp_model.double()

        lfm_state_dict = torch.load(filepath+'lfm.pt')
        lfm = cls(lfm_args[0], gp_model, *lfm_args[1:], **lfm_kwargs)
        lfm.load_state_dict(lfm_state_dict)
        return lfm
