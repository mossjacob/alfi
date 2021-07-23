import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.nn.functional import relu
from gpytorch.optim import NGD

from alfi.models import generate_multioutput_gp
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
    num_genes = 2000
    num_outputs = num_genes * 2
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

    gp_model = generate_multioutput_gp(num_latents, inducing_points,
                                       use_scale=False, initial_lengthscale=3,
                                       gp_kwargs=dict(natural=use_natural))

    y_target = data
    u_y = y_target[:num_genes]  # (num_genes, num_cells)
    s_y = y_target[num_genes:]  # (num_genes, num_cells)
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
    nonlinearity = softplus if params['nonlinearity'] == 'softplus' else relu

    nonzero_mask = list()
    for gene_index in range(num_outputs // 2):
        nonzero_mask.append(torch.logical_and(s_y[gene_index, :] > 0,
                                              u_y[gene_index, :] > 0))
    nonzero_mask = torch.stack(nonzero_mask).repeat(2, 1)

    lfm_kwargs = dict(
        nonlinearity=nonlinearity,
        decay_rate=gamma,
        num_training_points=num_cells,
        nonzero_mask=nonzero_mask
    )
    lfm = RNAVelocityLFM(num_outputs, gp_model, config, **lfm_kwargs)

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
    end_t = 12
    t_predict = torch.linspace(0, end_t, 80, dtype=torch.float32)
