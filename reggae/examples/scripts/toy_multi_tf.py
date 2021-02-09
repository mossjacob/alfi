
import torch
from reggae.utilities import save, load, is_cuda

from reggae.gp.variational.models import MultiLFM
from reggae.gp.variational.trainer import TranscriptionalTrainer
from reggae.data_loaders.datasets import ArtificialData
from reggae.plot.variational_plotters import Plotter
from matplotlib import pyplot as plt

import numpy as np

f64 = np.float64

#%%

dataset = ArtificialData()

num_genes = dataset.num_genes
num_tfs = dataset.num_tfs
gene_names = np.arange(num_genes)
t_inducing = torch.linspace(f64(0), f64(1), 7, dtype=torch.float64).reshape((-1, 1))
t_observed = dataset.t.view(-1)
print('Inducing points', t_inducing.shape)
print(gene_names.shape)

#%%

model = MultiLFM(num_genes, num_tfs, t_inducing, dataset, fixed_variance=None)
model = model.cuda() if is_cuda() else model

optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
trainer = TranscriptionalTrainer(model, optimizer, dataset)

print(t_observed.shape, dataset[0][1].shape)

#%% md

### Outputs prior to training:

#%%

t_predict = torch.linspace(f64(0), f64(1), 80)
rtol = 1e-3
atol = rtol/10

model_kwargs = {
    'rtol': rtol, 'atol': atol,
    'num_samples': 1
}

plotter = Plotter(model, gene_names, t_inducing)
plotter.plot_outputs(t_predict, t_scatter=t_observed, y_scatter=dataset[0][1].transpose(0,1), model_kwargs=model_kwargs)
plotter.plot_latents(t_predict)

#%%

for i in range(num_tfs):
    plt.plot(dataset.f_observed[i])

#%%

import time
start = time.time()
tol = 1e-4
output = trainer.train(20, rtol=tol, atol=tol/10, report_interval=1, plot_interval=2)
end = time.time()
print(end - start)



### Outputs after training

#%%

plotter = Plotter(model, gene_names, t_inducing)

tol = 1e-4
plotter.plot_losses(trainer, last_x=100)
plotter.plot_outputs(t_predict, t_scatter=t_observed, y_scatter=dataset[0][1].transpose(0,1), model_kwargs=model_kwargs)
plotter.plot_latents(t_predict, ylim=(-2, 9))


#%%

save(model, 'multitf')

#%%
