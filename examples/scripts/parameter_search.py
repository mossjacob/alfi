
import torch

from lafomo.variational.models import SingleLinearLFM
from lafomo.variational.trainer import Trainer
from lafomo.data_loaders import load_barenco_puma
import numpy as np

f64 = np.float64
replicate = 0

m_observed, f_observed, σ2_m_pre, σ2_f_pre, t = load_barenco_puma('../data/')

m_df, m_observed = m_observed
f_df, f_observed = f_observed
# Shape of m_observed = (replicates, genes, times)
m_observed = torch.tensor(m_observed)[replicate]

σ2_m_pre = f64(σ2_m_pre)

num_genes = m_observed.shape[0]
num_tfs = 1

t_inducing = torch.linspace(f64(0), f64(1), 5, dtype=torch.float64).reshape((-1, 1))
t_observed = torch.linspace(f64(0), f64(1), 7).view(-1)


def run_timepoints():
    for points in range(0, 5):
        start_loss = 0
        end_loss = 0
        num_averagings = 5
        for _ in range(num_averagings):  # average over random initialisations
            model = SingleLinearLFM(num_genes, num_tfs, t_inducing, t_observed, extra_points=points, known_variance=σ2_m_pre[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
            trainer = Trainer(model, optimizer, (t_observed, m_observed))
            tol = 1e-3
            # trainer = Trainer(optimizer)
            output = trainer.train(15, rtol=tol, atol=tol / 10, report_interval=16, plot_interval=16)

            start_loss += trainer.losses[1] / num_averagings
            end_loss += trainer.losses[-1] / num_averagings

        print(f'For {points} extra timepoints, the start and end loss are: {start_loss} and {end_loss} resp.')


def run_samples():
    for samples in range(0, 10):
        start_loss = 0
        end_loss = 0
        num_averagings = 5
        for _ in range(num_averagings):  # average over random initialisations
            model = SingleLinearLFM(num_genes, num_tfs, t_inducing, t_observed, known_variance=σ2_m_pre[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
            trainer = Trainer(model, optimizer, (t_observed, m_observed))
            tol = 1e-3
            # trainer = Trainer(optimizer)
            output = trainer.train(15, rtol=tol, atol=tol / 10, report_interval=16, plot_interval=16, num_samples=samples)

            start_loss += trainer.losses[1] / num_averagings
            end_loss += trainer.losses[-1] / num_averagings

        print(f'For {samples} samples, the start and end loss are: {start_loss} and {end_loss} resp.')


def run_inducing():
    for inducing_points in range(0, 10):
        start_loss = 0
        end_loss = 0
        num_averagings = 5
        for _ in range(num_averagings):  # average over random initialisations
            t_inducing = torch.linspace(f64(0), f64(1), inducing_points, dtype=torch.float64).reshape((-1, 1))
            model = SingleLinearLFM(num_genes, num_tfs, t_inducing, t_observed, known_variance=σ2_m_pre[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
            trainer = Trainer(model, optimizer, (t_observed, m_observed))
            tol = 1e-3
            # trainer = Trainer(optimizer)
            output = trainer.train(15, rtol=tol, atol=tol / 10, report_interval=16, plot_interval=16, num_samples=samples)

            start_loss += trainer.losses[1] / num_averagings
            end_loss += trainer.losses[-1] / num_averagings

        print(f'For {inducing_points} samples, the start and end loss are: {start_loss} and {end_loss} resp.')


if __name__ == '__main__':
    ## Firstly we go through how many extra timepoints the TF sampler should use
    run_timepoints()
    ## Secondly we go through how many likelihood samples the model should use
    run_samples()
