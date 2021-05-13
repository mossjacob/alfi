#%%

from torch.optim import Adam
from gpytorch.optim import NGD
from experiments.partial import build_partial, plot_partial
from pathlib import Path
import numpy as np

from lafomo.datasets import DrosophilaSpatialTranscriptomics, HomogeneousReactionDiffusion
from lafomo.trainers import PartialPreEstimator
from lafomo.plot import plot_spatiotemporal_data
from lafomo.plot.misc import plot_variational_dist
from lafomo.utilities.torch import spline_interpolate_gradient, softplus

from matplotlib import pyplot as plt
import torch
from lafomo.configuration import VariationalConfiguration

mrna_q2s = list()
mrna_cias = list()
protein_q2s = list()
protein_cias = list()

kni_params = dict(sensitivity=0.183,
                  decay=0.0770,
                  diffusion=0.0125)
gt_params = dict(sensitivity=0.1107,
                  decay=0.1110,
                  diffusion=0.0159)
kr_params = dict(sensitivity=0.0970,
                 decay=0.0764,
                 diffusion=0.0015)
params = dict(kr=kr_params, kni=kni_params, gt=gt_params)
gene = 'kni'
data = 'dros-kni'
dataset = DrosophilaSpatialTranscriptomics(
    gene=gene, data_dir='../../../data', scale=True)

params = dict(lengthscale=10,
              **params[gene],
              parameter_grad=False,
              warm_epochs=-1,
              natural=False,
              zero_mean=False,
              clamp=True)

lfm, trainer, plotter = build_partial(
    dataset,
    params)

sensitivity = (torch.tensor(params['sensitivity']))
decay = (torch.tensor(params['decay']))
diffusion = (torch.tensor(params['diffusion']))
tx = trainer.tx
num_t = tx[0, :].unique().shape[0]
num_x = tx[1, :].unique().shape[0]
y_target = trainer.y_target[0]
y_matrix = y_target.view(num_t, num_x)
print(y_matrix.shape)
dy_t = list()
for i in range(num_x):
    t = tx[0][::num_x]
    y = y_matrix[:, i].unsqueeze(-1)
    t_interpolate, y_interpolate, y_grad, _ = \
        spline_interpolate_gradient(t, y)
    dy_t.append(y_grad)
dy_t = torch.stack(dy_t)

# fig, axes = plt.subplots(nrows=2, figsize=(5, 7))
d2y_x = list()
dy_x = list()
for i in range(num_t):
    t = tx[1][:num_x]
    y = y_matrix[i].unsqueeze(-1)
    t_interpolate, y_interpolate, y_grad, y_grad_2 = \
        spline_interpolate_gradient(t, y)
    d2y_x.append(y_grad_2)
    # axes[0].plot(t_interpolate, y_interpolate)
    #
    # axes[1].plot(y_grad_2)
    dy_x.append(y_grad)

d2y_x = torch.stack(d2y_x)
dy_x = torch.stack(dy_x)[..., ::10, 0].reshape(1, -1)
d2y_x = d2y_x[..., ::10, 0].reshape(1, -1)
dy_t = dy_t[..., ::10, 0].t().reshape(1, -1)


def pde_func(y, u, sensitivity, decay, diffusion):
    # y (1, 1681) u (25, 1, 41, 41) s (25, 1)
    dy_t = (sensitivity * u.view(u.shape[0], -1) -
            decay * y.view(1, -1) +
            diffusion * d2y_x)
    return dy_t

orig_data = dataset.orig_data.squeeze().t()

for i in range(5):
    lfm, trainer, plotter = build_partial(
        dataset,
        params)


    ts = tx[0, :].unique().numpy()
    xs = tx[1, :].unique().numpy()
    extent = [ts[0], ts[-1], xs[0], xs[-1]]

    train_ratio = 0.3
    num_training = int(train_ratio * tx.shape[1])
    if params['natural']:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.05)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.09)]


    pre_estimator = PartialPreEstimator(
        lfm, optimizers, dataset, pde_func,
        input_pair=(trainer.tx, trainer.y_target), target=dy_t.t(),
        train_mask=trainer.train_mask
    )

    import time
    t0 = time.time()
    lfm.pretrain(True)
    lfm.config.num_samples = 50
    times = pre_estimator.train(100, report_interval=5)
    lfm.config.num_samples = 5

    trainer.plot_outputs = False
    lfm.pretrain(False)
    trainer.train(300, report_interval=10)

    from lafomo.plot import tight_kwargs
    plot_partial(dataset, lfm, trainer, plotter, Path('./'), params)

    plt.savefig(Path('./') / f'kinetics-{gene}-{i}.pdf', **tight_kwargs)

    # from lafomo.utilities.torch import q2, cia
    # f = lfm(tx)
    # f_mean = f.mean.detach()
    # f_var = f.variance.detach()
    # y_target = trainer.y_target[0]

    print('Run ', i)
    # protein_q2 = q2(y_target.squeeze(), f_mean.squeeze())
    protein_q2 = trainer.prot_q2_best
    # protein_cia = cia(y_target.squeeze(), f_mean.squeeze(), f_var.squeeze())
    protein_cia = trainer.cia[1]
    print('Protein Q2', protein_q2)
    print('Protein CA', protein_cia)
    protein_q2s.append(protein_q2.item())
    protein_cias.append(protein_cia.item())

    # gp = lfm.gp_model(tx.t())
    # lf_target = orig_data[trainer.t_sorted, 2]
    # f_mean = gp.mean.detach()
    # f_var = gp.variance.detach()

    # mrna_q2 = q2(lf_target.squeeze(), f_mean.squeeze())
    mrna_q2 = trainer.mrna_q2_best
    # mrna_cia = cia(lf_target.squeeze(), f_mean.squeeze(), f_var.squeeze())
    mrna_cia = trainer.cia[0]
    print('mRNA Q2', mrna_q2)
    print('mRNA CA', mrna_cia)
    mrna_q2s.append(mrna_q2.item())
    mrna_cias.append(mrna_cia.item())

print(protein_q2s)
protein_q2s = torch.tensor(protein_q2s)
protein_cias = torch.tensor(protein_cias)
mrna_q2s = torch.tensor(mrna_q2s)
mrna_cias = torch.tensor(mrna_cias)
print('Final')
print(f'Protein Q2: {protein_q2s.mean(0):.04f} pm {protein_q2s.std(0):.04f}')
print(f'Protein CIA: {protein_cias.mean(0):.04f} pm {protein_cias.std(0):.04f}')
print(f'mRNA Q2: {mrna_q2s.mean(0):.04f} pm {mrna_q2s.std(0):.04f}')
print(f'mRNA CIA: {mrna_cias.mean(0):.04f} pm {mrna_cias.std(0):.04f}')
