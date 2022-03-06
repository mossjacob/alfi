#%%
from experiments.partial import build_partial, plot_partial, pretrain_partial

from alfi.datasets import DrosophilaSpatialTranscriptomics, HomogeneousReactionDiffusion
from alfi.models import TrainMode
from matplotlib import pyplot as plt
import torch
from alfi.configuration import VariationalConfiguration

mrna_q2s = list()
mrna_cias = list()
protein_q2s = list()
protein_cias = list()

gene = 'kni'

dataset = DrosophilaSpatialTranscriptomics(
    gene=gene, data_dir='./data', scale=True)
disc = dataset.disc

params = {gene: {}}

params = dict(lengthscale=10,
              **params[gene],
              parameter_grad=False,
              warm_epochs=-1,
              natural=True,
              zero_mean=True,
              clamp=True)

lfm, trainer, plotter = build_partial(
    dataset,
    params)

tx = trainer.tx
orig_data = dataset.orig_data.squeeze().t()
num_t = tx[0, :].unique().shape[0]
num_x = tx[1, :].unique().shape[0]

num_t_orig = orig_data[:, 0].unique().shape[0]
num_x_orig = orig_data[:, 1].unique().shape[0]

y_target = trainer.y_target[0]
y_matrix = y_target.view(num_t_orig, num_x_orig)

for i in range(1):
    lfm, trainer, plotter = build_partial(
        dataset,
        params)


    ts = tx[0, :].unique().numpy()
    xs = tx[1, :].unique().numpy()
    extent = [ts[0], ts[-1], xs[0], xs[-1]]

    times = pretrain_partial(dataset, lfm, trainer, params)

    trainer.plot_outputs = False
    lfm.set_mode(TrainMode.NORMAL)
    trainer.train(150, report_interval=10)
    lfm.save(f'./{gene}{i}')
    # from alfi.plot import tight_kwargs
    # plot_partial(dataset, lfm, trainer, plotter, Path('./'), params)
    #
    # plt.savefig(Path('./') / f'kinetics-{gene}-{i}.pdf', **tight_kwargs)

    # from alfi.utilities.torch import q2, cia
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

protein_q2s = torch.tensor(protein_q2s)
protein_cias = torch.tensor(protein_cias)
mrna_q2s = torch.tensor(mrna_q2s)
mrna_cias = torch.tensor(mrna_cias)
print('Final')

print(f'Protein Q2: {protein_q2s.mean(0):.04f} pm {protein_q2s.std(0):.04f}')
print(f'Protein CIA: {protein_cias.mean(0):.04f} pm {protein_cias.std(0):.04f}')
print(f'mRNA Q2: {mrna_q2s.mean(0):.04f} pm {mrna_q2s.std(0):.04f}')
print(f'mRNA CIA: {mrna_cias.mean(0):.04f} pm {mrna_cias.std(0):.04f}')

torch.save(protein_q2s, f'experiments/{gene}_protein_q2')
torch.save(protein_cias, f'experiments/{gene}_protein_cia')
torch.save(mrna_q2s, f'experiments/{gene}_mrna_q2')
torch.save(mrna_cias, f'experiments/{gene}_mrna_cia')
