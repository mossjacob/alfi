from abc import ABC
from enum import Enum

import torch

from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from .lfm import LFM
from alfi.configuration import VariationalConfiguration
from alfi.mlls import MaskedVariationalELBO
from alfi.utilities import is_cuda


class TrainMode(Enum):
    NORMAL = 0
    """
    Gradient matching: the derivative of the spline interpolation is matched with the output of the ODE function
    """
    GRADIENT_MATCH = 1
    """
    Filter: the output of the ODE function is matched with the filtered data bucketed and filtered into the same
    timepoints
    """
    FILTER = 2
    """
    Samples: return samples from the ODE
    """
    SAMPLES = 3


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
        self.train_mode = TrainMode.NORMAL
        self.config = config
        self.dtype = dtype

        try:
            self.inducing_points = self.gp_model.get_inducing_points()
        except AttributeError:
            raise AttributeError('The GP model must define a function `get_inducing_points`.')

        # Construct likelihood
        self.num_tasks = num_outputs
        if num_training_points is None:
            num_training_points = self.inducing_points.numel()  # TODO num_data refers to the number of training datapoints

        self.num_latents = gp_model.variational_strategy.num_tasks
        if config.latent_data_present:  # add latent force likelihood
            self.num_tasks += self.num_latents

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)

        self.loss_fn = MaskedVariationalELBO(self.likelihood, gp_model, num_training_points, combine_terms=False)

    def nonvariational_parameters(self):
        variational_keys = dict(self.gp_model.named_variational_parameters()).keys()
        named_parameters = dict(self.named_parameters())
        return [named_parameters[key] for key in named_parameters.keys()
                if key[len('gp_model.'):] not in variational_keys]

    def variational_parameters(self):
        return self.gp_model.variational_parameters()

    def summarise_gp_hyp(self):
        with torch.no_grad():
            def convert(x):
                inducing_points = self.inducing_points.cuda() if is_cuda() else self.inducing_points
                x = x.detach().cpu().view(-1).numpy()[:5]
                # noise = self.gp_model(inducing_points).variance.mean(dim=0).cpu()
                return str(x)  #+ ' noise: ' + str(noise)
            if self.gp_model.covar_module.lengthscale is not None:
                return convert(self.gp_model.covar_module.lengthscale)
            elif hasattr(self.gp_model.covar_module, 'base_kernel'):
                kernel = self.gp_model.covar_module.base_kernel
                if hasattr(kernel, 'kernels'):
                    if hasattr(kernel.kernels[0], 'lengthscale'):
                        return convert(kernel.kernels[0].lengthscale)
                else:
                    return convert(self.gp_model.covar_module.base_kernel.lengthscale)
            else:
                return ''

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.gp_model.train(mode)
        self.likelihood.train(mode)

    def set_mode(self, mode=TrainMode.NORMAL):
        self.train_mode = mode

    def eval(self):
        self.gp_model.eval()
        self.likelihood.eval()
        if self.train_mode == TrainMode.GRADIENT_MATCH or self.train_mode == TrainMode.FILTER:
            self.set_mode(TrainMode.NORMAL)

    def predict_m(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Calls self on input `t_predict`
        """
        return self(t_predict.view(-1), **kwargs)

    def predict_f(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Returns the latents
        """
        self.eval()
        with torch.no_grad():
            q_f = self.gp_model(t_predict)
        self.train()
        return q_f

    def save(self, filepath):
        torch.save(self.gp_model.state_dict(), str(filepath)+'gp.pt')
        torch.save(self.state_dict(), str(filepath)+'lfm.pt')

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
