
import torch
import numpy as np

from torch.nn import Parameter
from gpytorch.optim import NGD
from torch.optim import Adam
from matplotlib import pyplot as plt
from gpytorch.constraints import Positive, Interval
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, MultiOutputGP, generate_multioutput_rbf_gp
from alfi.plot import Plotter1d, Colours, tight_kwargs
from alfi.trainers import VariationalTrainer, PreEstimator
from alfi.utilities.data import hafner_ground_truth
from alfi.datasets import HafnerData
from alfi.utilities.torch import softplus
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
            sens = torch.tensor(4.2)
            dec = torch.tensor(0.21)
            self.lfm.raw_sensitivity[6] = self.lfm.positivity.inverse_transform(sens)
            self.lfm.raw_decay[6] = self.lfm.positivity.inverse_transform(dec)
        super().after_epoch()


class SoftplusTranscriptionLFM(TranscriptionLFM):
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.positivity = Positive()
        # self.decay_constraint = Interval(0, 1)
        self.raw_decay = Parameter(0.1 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
        self.raw_basal = Parameter(torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
        self.raw_sensitivity = Parameter(8 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))

    def mix(self, f):
        # I = 1 so just repeat for num_outputs
        return softplus(f).repeat(1, self.num_outputs, 1)

    def predict_f(self, t_predict, **kwargs):
        # Sample from the latent distribution
        q_f = super().predict_f(t_predict)
        f = q_f.sample(torch.Size([500])).permute(0, 2, 1)  # (S, I, T)
        print(f.shape)
        # This is a hack to wrap the latent function with the nonlinearity. Note we use the same variance.
        f = torch.mean(self.mix(f), dim=0)[0].unsqueeze(0)
        print(f.shape, q_f.mean.shape, q_f.scale_tril.shape)
        batch_mvn = MultivariateNormal(f, q_f.covariance_matrix.unsqueeze(0))
        print(batch_mvn)
        return MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)


dataset = HafnerData(replicate=0, data_dir='./data/', extra_targets=False)
ground_truth = hafner_ground_truth()
num_inducing = 12
use_natural = False
num_times = dataset[0][0].shape[0]
num_replicates = 1
num_genes = len(dataset.gene_names)
num_tfs = 1
t_inducing = torch.linspace(0, 12, num_times, dtype=torch.float64)
t_observed = torch.linspace(0, 12, num_times)
t_predict = torch.linspace(0, 15, 80, dtype=torch.float32)

config = VariationalConfiguration(
    num_samples=70,
    initial_conditions=False
)

inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
step_size = 5e-1
num_training = dataset.m_observed.shape[-1]
gp_model = generate_multioutput_rbf_gp(num_tfs, inducing_points,
                                       zero_mean=False,
                                       initial_lengthscale=2,
                                       gp_kwargs=dict(natural=use_natural))
lfm = TranscriptionLFM(num_genes, gp_model, config, num_training_points=num_training)
plotter = Plotter1d(lfm, dataset.gene_names, style='seaborn')

track_parameters = [
    'raw_basal',
    'raw_decay',
    'raw_sensitivity',
    'gp_model.covar_module.raw_lengthscale',
]
if use_natural:
    variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
    parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.02)
    optimizers = [variational_optimizer, parameter_optimizer]
else:
    optimizers = [Adam(lfm.parameters(), lr=0.05)]


trainer = VariationalTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)


lfm.train()
# trainer = Trainer(optimizer)
output = trainer.train(500, step_size=5e-1, report_interval=10)

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)
t_predict = torch.linspace(0, 15, 80, dtype=torch.float32)

lfm.eval()
q_m = lfm.predict_m(t_predict, step_size=1e-1)
q_f = lfm.predict_f(t_predict)

plotter.plot_losses(trainer, last_x=200)
nrows = 3
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))

row = col = 0
for i in range(8):
    if i == (row+1) * 3:
        row += 1
        col = 0
    ax = axes[row, col]
    plotter.plot_gp(q_m, t_predict, replicate=0, ax=ax,# ylim=(-2, 25.2),
                    color=Colours.line_color, shade_color=Colours.shade_color,
                    t_scatter=dataset.t_observed, y_scatter=dataset.m_observed,
                    only_plot_index=i, num_samples=0)
    col += 1
    ax.set_title(dataset.gene_names[i])
plotter.plot_gp(q_f, t_predict, ax=axes[nrows-1, ncols-1],
                # ylim=(-1, 5),
                transform=softplus,
                num_samples=3,
                color=Colours.line2_color,
                shade_color=Colours.shade2_color)
axes[nrows-1, ncols-1].set_title('Latent force (p53)')
for col in range(ncols):
    axes[nrows-1, col].set_xlabel('Time (h)')

plt.tight_layout()
plt.savefig('./experiments/hafner-multi.pdf', **tight_kwargs)


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 2.9),
                         gridspec_kw=dict(width_ratios=[1, 1, 0.5, 1.9], wspace=0, hspace=0.75))
row = col = 0
plots = [2, 4, 7, 6]
lbs = [2, 0, 2, 5]
for i in range(4):
    if i == (row+1) * 2:
        row += 1
        col = 0
    ax = axes[row, col]
    plotter.plot_gp(q_m, t_predict, replicate=0, ax=ax,# ylim=(-2, 25.2),
                    color=Colours.line_color, shade_color=Colours.shade_color,
                    t_scatter=dataset.t_observed, y_scatter=dataset.m_observed,
                    only_plot_index=plots[i], num_samples=0)
    ax.set_title(dataset.gene_names[plots[i]])
    mean = q_m.mean.detach().transpose(0, 1)[plots[i]]
    lb = torch.floor(mean.min()-1)
    ub = torch.ceil(mean.max() + 1)
    brange = ub - lb
    ub = int(lb + brange*1.1)
    lb = lbs[i]
    ax.set_ylim(lb, ub)
    ax.set_xlim(0, 15)
    ax.set_yticks([lb, ub])
    ax.figure.subplotpars.wspace = 0
    if col > 0:
        ax.set_yticks([])
        ax.set_xticks([5, 10, 15])

    col += 1
y = [1.5, 4.8, 13.7, 5, 2, 1.4, 3.2, 4, 1.4, 1.5]
# y = y/np.mean(y)*np.mean(p) * 1.75-0.16
# y = scaler.fit_transform(np.expand_dims(y, 0))
axes[0, 3].plot(np.linspace(0, 10, len(y)), y, color=Colours.line2_color)
axes[0, 3].set_ylim(0, 15)
axes[0, 3].set_ylabel('Fold change')
axes[0, 3].set_yticks([0.0, 15.0])
axes[0, 3].set_title('Western blot (Hafner et al., 2017)')
t_temp = torch.linspace(0, 10, 80, dtype=torch.float32)
q_f = lfm.predict_f(t_temp)
plotter.plot_gp(q_f, t_temp, ax=axes[1, 3],
                transform=softplus,
                num_samples=3,
                color=Colours.line2_color,
                shade_color=Colours.shade2_color)
axes[1, 3].set_yticks([0, 4])
for i in range(2):
    axes[i, 3].set_xlim(0, 10)
    axes[i, 2].set_visible(False)

axes[1, 3].set_title('Latent force inference (ours)')
axes[1, 3].set_xlabel('Time (h)')
axes[1, 3].set_ylabel('FPKM $\ell_2$')
axes[1, 3].set_yticks([0.0, 6.0])
axes[1, 3].set_ylim(0.0, 6)
# axes[1, 3].yaxis.set_major_formatter(FormatStrFormatter('  %.0f'))

fig.tight_layout()

plt.savefig('./experiments/hafner-combined.pdf', **tight_kwargs)

labels = ['Basal rates', 'Sensitivities', 'Decay rates']
keys = ['raw_basal', 'raw_sensitivity', 'raw_decay']
constraints = [lfm.positivity, lfm.positivity, lfm.positivity]
kinetics = list()
for i, key in enumerate(keys):
    kinetics.append(
        constraints[i].transform(trainer.parameter_trace[key][-1].squeeze()).numpy())

kinetics = np.array(kinetics)
plotter.plot_double_bar(kinetics, titles=labels,
                        figsize=(9, 3),
                        ground_truths=hafner_ground_truth())
                        # yticks=[
                        #     np.linspace(0, 0.12, 5),
                        #     np.linspace(0, 1.2, 4),
                        #     np.arange(0, 1.1, 0.2),
                        # ])
plt.tight_layout()
# plt.savefig('./hafner-kinetics.pdf', **tight_kwargs)

# plotter.plot_convergence(trainer)


# print(lfm.decay_rate)
print(lfm.sensitivity[6])
for i in range(20):
    plt.plot(lfm.positivity.transform(torch.stack(trainer.parameter_trace['raw_sensitivity'])[:, i]))
