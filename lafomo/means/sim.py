import gpytorch
import torch

from gpytorch.constraints import Positive


class SIMMean(gpytorch.means.Mean):

    def __init__(self, covar_module, num_genes, initial_basal):
        super().__init__()
        self.covar_module = covar_module
        self.pos_contraint = Positive()
        self.covar_module = covar_module
        self.num_genes = num_genes

        self.register_parameter(
            name='raw_basal', parameter=torch.nn.Parameter(
                self.pos_contraint.inverse_transform(0.05 * torch.ones(self.num_genes)))
        )
        self.register_constraint("raw_basal", self.pos_contraint)

        # self.basal = initial_basal

    @property
    def basal(self):
        return self.pos_contraint.transform(self.raw_basal)

    @basal.setter
    def basal(self, value):
        self.initialize(raw_basal=self.pos_contraint.inverse_transform(value))

    def forward(self, x):
        block_size = int(x.shape[0] / self.num_genes)
        m = (self.basal / self.covar_module.decay).view(-1, 1)
        m = m.repeat(1, block_size).view(-1)
        return m
