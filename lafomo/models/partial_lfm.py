import torch

from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from torch_fenics import FEniCSModule

from lafomo.models import VariationalLFM
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import softplus


class PartialLFM(VariationalLFM):
    def __init__(self,
                 num_outputs,
                 gp_model: ApproximateGP,
                 fenics_model: FEniCSModule,
                 fenics_parameters: list,
                 config: VariationalConfiguration,
                 num_training_points=None,
                 dtype=torch.float64):
        super().__init__(num_outputs, gp_model, config, num_training_points, dtype)
        if self.config.initial_conditions:
            raise Exception('Initial conditions are not implemented for PartialLFM.')

        self.time_steps = fenics_model.time_steps
        self.mesh_cells = fenics_model.mesh.cells().shape[0]
        self.fenics_module = fenics_model
        self.fenics_parameters = fenics_parameters
        self.fenics_named_parameters = dict()
        name = 0
        for parameter in self.fenics_parameters:
            self.register_parameter(name='fenics' + str(name), param=parameter)
            self.fenics_named_parameters['fenics' + str(name)] = parameter
            name += 1

    def forward(self, tx, step_size=1e-1, return_samples=False, **kwargs):
        """
        tx : torch.Tensor
            Shape (2, num_times) or, if in pretrain mode, then a tuple containing input and output
        h : torch.Tensor the initial state of the ODE
            Shape (num_genes, 1)
        Returns
        -------
        Returns evolved h across times t.
        Shape (num_genes, num_points).
        """
        self.nfe = 0

        # Get GP outputs
        if self.pretrain_mode:
            t_f = tx[0].transpose(0, 1)
            data = tx[0]
        else:
            t_f = tx.transpose(0, 1)
            data = tx
        num_t = data[0, :].unique().shape[0]
        num_x = data[1, :].unique().shape[0]

        q_u = self.gp_model(t_f)
        u = q_u.rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)
        u = self.G(u)  # (S, num_outputs, tx)
        u = u.view(*u.shape[:2], num_t, num_x)

        if self.pretrain_mode:
            params = [softplus(param.repeat(self.config.num_samples, 1)) for param in self.fenics_parameters]
            outputs = kwargs['pde_func'](tx[1], u, *params)
        else:
            outputs = self.solve_pde(u)

        if return_samples:
            return outputs

        f_mean = outputs.mean(dim=0).view(1, -1)  # shape (batch, times * distance)
        f_var = outputs.var(dim=0).view(1, -1) + 1e-7
        # TODO: make distribution something less constraining
        f_covar = torch.diag_embed(f_var)
        batch_mvn = MultivariateNormal(f_mean, f_covar)
        return MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def solve_pde(self, u):
        """

        @param u: Shape (S, 1, num_t, num_x)
        @return:
        """
        # Integrate forward from the initial positions h0.
        outputs = list()
        y_prev = torch.zeros((self.config.num_samples, self.mesh_cells + 1), requires_grad=False, dtype=torch.float64)

        # print('yprev u', y_prev.shape, u.shape)

        # t = df['t'].values[:41]
        for n in range(self.time_steps + 1):
            u_n = u[:, 0, n]  # (S, t)

            params = [softplus(param.repeat(self.config.num_samples, 1)) for param in self.fenics_parameters]

            y_prev = self.fenics_module(y_prev, u_n, *params)

            # y_prev shape (N, 21)
            outputs.append(y_prev)

        outputs = torch.stack(outputs).permute(1, 0, 2) # (S, T, X)
        return outputs

    def G(self, u):
        return u
