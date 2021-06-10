import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD

from alfi.models import generate_multioutput_rbf_gp
from alfi.plot import Plotter1d
from alfi.utilities.torch import softplus
from alfi.datasets import SingleCellKidney, Pancreas
from alfi.impl.odes import RNAVelocityLFM, RNAVelocityConfiguration
from alfi.impl.trainers import EMTrainer

def build_rnavelocity(dataset, params, **kwargs):
    data = dataset.m_observed.squeeze()
    num_cells = dataset[0].shape[1]
    num_latents = 10
    num_inducing = 50  # (I x m x 1)
    end_t = 12
    use_natural = False

    config = RNAVelocityConfiguration(
        num_samples=30,
        num_cells=num_cells,
        end_pseudotime=end_t
    )

    print('Number of cells:', num_cells)
    print('Number of latent GPs (# transcription rates):', num_latents)

    step_size = 1e-1

    inducing_points = torch.linspace(0, end_t, num_inducing).repeat(num_latents, 1).view(num_latents, num_inducing, 1)

    gp_model = generate_multioutput_rbf_gp(num_latents, inducing_points,
                                           use_scale=False, initial_lengthscale=3,
                                           gp_kwargs=dict(natural=use_natural))

    y_target = data
    u_y = y_target[:2000]  # (num_genes, num_cells)
    s_y = y_target[2000:]  # (num_genes, num_cells)
    x = s_y > torch.tensor(np.percentile(s_y, 98, axis=1)).unsqueeze(-1)
    s = s_y * x
    x = u_y > torch.tensor(np.percentile(u_y, 98, axis=1)).unsqueeze(-1)
    u = u_y * x

    s = s.unsqueeze(-1)
    u = u.unsqueeze(1)
    gamma = torch.matmul(u, s).squeeze()
    l2 = s.squeeze().square().sum(dim=1)
    gamma /= l2
    gamma = gamma.unsqueeze(-1)

    lfm = RNAVelocityLFM(4000, gp_model, config, nonlinearity=softplus, decay_rate=gamma, num_training_points=num_cells)
    if use_natural:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_cells, lr=0.05)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.02)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [torch.optim.Adam(lfm.parameters(), lr=0.06)]
    trainer = EMTrainer(lfm, optimizers, dataset, batch_size=4000)
    plotter = Plotter1d(lfm, dataset.gene_names)

    return lfm, trainer, plotter


def plot_rnavelocity(dataset, lfm, trainer, plotter, filepath, params):
    cpe_index = np.where(dataset.loom.var.index == 'Cpe')[0][0]
    t_predict = torch.linspace(0, end_t, 80, dtype=torch.float32)
