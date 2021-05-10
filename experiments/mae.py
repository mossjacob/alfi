import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
from gpytorch.constraints import Positive
import seaborn as sns
import numpy as np

from lafomo.datasets import P53Data, ToyTranscriptomicGenerator
from lafomo.configuration import VariationalConfiguration
from lafomo.models import OrdinaryLFM, generate_multioutput_rbf_gp
from lafomo.trainers import VariationalTrainer
from .variational import TranscriptionLFM
from lafomo.utilities.data import p53_ground_truth
from torch.nn.functional import l1_loss
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from lafomo.models import ExactLFM
from lafomo.trainers import ExactTrainer


""" Experiment for finding mean absolute errors """
def get_datasets():
    from lafomo.datasets import LFMDataset
    class DSet(LFMDataset):
        def __init__(self, disc, t_obs, m_obs, f_obs):
            self.m_observed = m_obs[::disc]
            self.m_observed_highres = m_obs

            self.f_observed = f_obs[::disc]
            self.f_observed_highres = f_obs

            self.t_observed = t_obs[::disc]
            self.t_observed_highres = t_obs

    datasets = list()
    for i in range(10):
        dataset_dict = torch.load(f'./dataset{num_success}.pt')
        dataset = DSet(
            dataset_dict['high_disc'],
            dataset_dict['t_obs'],
            dataset_dict['m_obs'],
            dataset_dict['f_obs'])
        print(dataset.m_observed)
        datasets.append(dataset)
    return datasets

def save_dataset(dataset, path):
    torch.save({
        'm_obs': dataset.m_observed_highres, 'high_disc': 10,
        't_obs': dataset.t_observed_highres,
        'f_obs': dataset.f_observed_highres,
    }, path)


def param_l1(lfm, ground_truth):
    if isinstance(lfm, ExactLFM):
        basal = lfm.mean_module.basal
        sensitivity = lfm.covar_module.sensitivity
        decay = lfm.covar_module.decay
    else:
        basal = lfm.basal_rate.squeeze()
        sensitivity = lfm.sensitivity.squeeze()
        decay = lfm.decay_rate.squeeze()
    print(basal.shape, ground_truth[0].shape)

    return (
        l1_loss(basal, torch.from_numpy(ground_truth[0])) +
        l1_loss(sensitivity, torch.from_numpy(ground_truth[1])) +
        l1_loss(decay, torch.from_numpy(ground_truth[2]))
    ).mean().item()

def train(lfm, trainer, m_targ, f_targ):
    f_maes = list()
    m_maes = list()
    i = 0
    for i in range(200 // 10):
        try:
            trainer.train(epochs=10, report_interval=50)
        except:
            break
        m_pred = lfm.predict_m(t_predict, jitter=1e-3)
        f_pred = lfm.predict_f(t_predict, jitter=1e-3)
        m_mae = l1_loss(m_pred.mean, m_targ).mean().item()
        f_mae = l1_loss(f_pred.mean, f_targ).mean().item()
        f_maes.append(f_mae)
        m_maes.append(m_mae)

    if i < 10:
        print('except')
        raise Exception('too few epochs')
    f_maes = torch.tensor(f_maes)
    m_maes = torch.tensor(m_maes)
    min_index = (f_maes + m_maes).argmin(dim=0)
    f_mae = f_maes[min_index]
    m_mae = m_maes[min_index]
    print('chosen', f_mae, m_mae)
    return m_mae, f_mae

with open('experiments/mae.txt', 'w') as f:
    num_genes = 5
    num_tfs = 1
    config = VariationalConfiguration(
        num_samples=80,
        initial_conditions=False
    )
    maes = list()
    maes2 = list()
    num_success = 0
    for i in range(20):
        if num_success >= 10:
            break
        print('Running ', i)
        dataset = ToyTranscriptomicGenerator(
            num_outputs=5,
            num_latents=1,
            num_times=10,
            softplus=False,
            dtype=torch.float32)
        dataset.generate_single(lengthscale=1.3)
        dataset.variance = 1e-3 * torch.ones(dataset.m_observed.shape[-1], dtype=torch.float32)
        start_time = dataset.t_observed[0]
        end_time = dataset.t_observed[-1]
        t_predict = torch.linspace(0, end_time, dataset.m_observed_highres.shape[-1], dtype=torch.float32)
        ground_truth = [
            dataset.lfm.basal_rate.detach().view(-1).numpy(),
            dataset.lfm.sensitivity.detach().view(-1).numpy(),
            dataset.lfm.decay_rate.detach().view(-1).numpy()
        ]
        m_targ = dataset.m_observed_highres.squeeze().t()
        f_targ = dataset.f_observed_highres.squeeze(0).t()

        #### Train Exact
        exact_lfm = ExactLFM(dataset, dataset.variance.reshape(-1))
        optimizer = torch.optim.Adam(exact_lfm.parameters(), lr=0.01)

        loss_fn = ExactMarginalLogLikelihood(exact_lfm.likelihood, exact_lfm)

        exact_trainer = ExactTrainer(
            exact_lfm, [optimizer], dataset, loss_fn=loss_fn,
            track_parameters=[
                'mean_module.raw_basal',
                'covar_module.raw_decay',
                'covar_module.raw_sensitivity',
                'covar_module.raw_lengthscale'
            ])

        exact_lfm.train()
        exact_lfm.likelihood.train()

        try:
            m_mae_exact, f_mae_exact = train(exact_lfm, exact_trainer, m_targ, f_targ)
        except Exception as e:
            print('Exception:', e)
            # raise e
            continue

        #### Train variational
        num_inducing = 12  # (I x m x 1)
        inducing_points = torch.linspace(0, end_time, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
        gp_model = generate_multioutput_rbf_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=True))

        lfm = TranscriptionLFM(num_genes, gp_model, config)

        track_parameters = [
            'raw_basal',
            'raw_decay',
            'raw_sensitivity',
            'gp_model.covar_module.raw_lengthscale',
        ]
        num_training = dataset.m_observed.shape[-1]

        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
        optimizers = [variational_optimizer, parameter_optimizer]
        trainer = P53ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)

        lfm.train()
        try:
            trainer.train(350, report_interval=10, step_size=1e-1)
        except:
            continue
        last_loss = trainer.losses[-1].sum()

        q_m_vari = lfm.predict_m(t_predict, step_size=1e-1)
        q_f_vari = lfm.predict_f(t_predict)

        m_mae_varia = l1_loss(q_m_vari.mean, m_targ).mean().item()
        f_mae_varia = l1_loss(q_f_vari.mean, f_targ).mean().item()
        param_mae_exact = param_l1(exact_lfm, ground_truth)
        param_mae_varia = param_l1(lfm, ground_truth)
        print(m_mae_exact, m_mae_varia, f_mae_exact, f_mae_varia)
        maes.append(np.stack([m_mae_exact, m_mae_varia, f_mae_exact, f_mae_varia]))

        f.write(f'{i}\t{m_mae_exact}\t{m_mae_varia}\t{f_mae_exact}\t{f_mae_varia}\n')
        save_dataset(dataset, f'./dataset{num_success}.pt')
        num_success += 1
    maes = np.array(maes)
    np.save('./maes.npy', maes)
    print(maes)
    print(maes.shape)

    print('avg', maes.mean(axis=0), maes.std(axis=0))
