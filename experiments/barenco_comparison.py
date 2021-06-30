import torch
from gpytorch.optim import NGD
from torch.optim import Adam
from matplotlib import pyplot as plt
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch.nn.functional import l1_loss

from alfi.datasets import P53Data
from alfi.configuration import VariationalConfiguration
from alfi.models import generate_multioutput_gp
from alfi.plot import Plotter1d, Colours, tight_kwargs
from alfi.trainers import VariationalTrainer, PreEstimator
from alfi.models import ExactLFM
from alfi.trainers import ExactTrainer
from experiments.variational import TranscriptionLFM


plt.rcParams.update({'font.size': 82})
plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


class ConstrainedTrainer(VariationalTrainer):
    def after_epoch(self):
        with torch.no_grad():
            sens = torch.tensor(1.)
            dec = torch.tensor(0.8)
            self.lfm.raw_sensitivity[3] = self.lfm.positivity.inverse_transform(sens)
            self.lfm.raw_decay[3] = self.lfm.positivity.inverse_transform(dec)
        super().after_epoch()


dataset = P53Data(replicate=0, data_dir='./data')
num_inducing = 20

num_replicates = 1
num_genes = len(dataset.gene_names)
num_tfs = 1
num_times = dataset[0][0].shape[0]
t_end = dataset.t_observed[-1]
use_natural = True

config = VariationalConfiguration(
    # preprocessing_variance=dataset.variance,
    num_samples=80,
    initial_conditions=False
)

inducing_points = torch.linspace(0, t_end, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
t_predict = torch.linspace(0, t_end+3, 80, dtype=torch.float32)
step_size = 5e-1
num_training = dataset.m_observed.shape[-1]
gp_model = generate_multioutput_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=use_natural))

lfm = TranscriptionLFM(num_genes, gp_model, config,
                       initial_basal=0.1,
                       initial_decay=0.1,
                       initial_sensitivity=2,
                       num_training_points=num_training)

plotter = Plotter1d(lfm, dataset.gene_names, style='seaborn')

track_parameters = [
    'raw_basal',
    'raw_decay',
    'raw_sensitivity',
    'gp_model.covar_module.raw_lengthscale',
]
if use_natural:
    variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.09)
    parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.02)
    optimizers = [variational_optimizer, parameter_optimizer]
    pre_variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
    pre_parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.005)
    pre_optimizers = [pre_variational_optimizer, pre_parameter_optimizer]

else:
    optimizers = [Adam(lfm.parameters(), lr=0.05)]
    pre_optimizers = [Adam(lfm.parameters(), lr=0.05)]

trainer = ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)
pre_estimator = PreEstimator(lfm, pre_optimizers, dataset, track_parameters=track_parameters)

lfm.loss_fn.num_data = num_training
trainer.train(200, report_interval=10, step_size=step_size);

q_m = lfm.predict_m(t_predict, step_size=1e-1)
q_f = lfm.predict_f(t_predict)


exact_lfm = ExactLFM(dataset, dataset.variance.reshape(-1))
optimizer = torch.optim.Adam(exact_lfm.parameters(), lr=0.07)

loss_fn = ExactMarginalLogLikelihood(exact_lfm.likelihood, exact_lfm)

track_parameters = [
    'mean_module.raw_basal',
    'covar_module.raw_decay',
    'covar_module.raw_sensitivity',
    'covar_module.raw_lengthscale',
]
exact_trainer = ExactTrainer(exact_lfm, [optimizer], dataset, loss_fn=loss_fn, track_parameters=track_parameters)
exact_plotter = Plotter1d(exact_lfm, dataset.gene_names)


exact_lfm.train()
exact_lfm.likelihood.train()
exact_trainer.train(epochs=150, report_interval=10);

exact_q_m = exact_lfm.predict_m(t_predict)
exact_q_f = exact_lfm.predict_f(t_predict)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 2.9),
                         gridspec_kw=dict(width_ratios=[1, 1, 0.3, 1.8], wspace=0, hspace=0.75))

row = 0
col = 0
ub = [3.5] * 4
for i in range(4):
    if i == 2:
        row += 1
        col = 0
    ax = axes[row, col]
    plotter.plot_gp(q_m, t_predict, replicate=0, ax=ax,
                    color=Colours.line_color, shade_color=Colours.shade_color,
                    t_scatter=dataset.t_observed, y_scatter=dataset.m_observed,
                    num_samples=0, only_plot_index=i)
    ax.set_ylim([-0.2, ub[i]])
    ax.set_yticks([0, 3])
    ax.set_title(dataset.gene_names[i])
    ax.set_xlim(-0.4, 15)
    if col > 0:
        ax.set_yticks([])
        ax.set_xticks([5, 10, 15])
    else:
        ax.set_xticks([0, 5, 10, 15])
    col += 1
plotter.plot_gp(q_f, t_predict, ax=axes[1, 3],
                ylim=(-1, 3.2),
                num_samples=0,
                color=Colours.line2_color,
                shade_color=Colours.shade2_color)
plotter.plot_gp(exact_q_f, t_predict, ax=axes[0, 3],
                ylim=(-1, 3.2), color=Colours.line2_color,
                shade_color=Colours.shade2_color)
titles = ['(Lawrence et al., 2007)', 'inference (ours)']
for i in range(2):
    axes[i, 3].set_title(f'Latent force {titles[i]}')
    axes[i, 3].set_yticks([-1, 3])
    axes[i, 3].set_xlim(0, 15)
    axes[i, 3].set_xticks([0, 5, 10, 15])
    axes[i, 2].set_visible(False)
axes[1, 3].set_xlabel('Time (h)')

plt.savefig('./experiments/barenco-combined.pdf', **tight_kwargs)

print(l1_loss(q_f.mean, exact_q_f.mean).item())
