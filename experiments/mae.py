import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
import numpy as np

from alfi.datasets import P53Data, ToyTranscriptomicGenerator
from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, generate_multioutput_gp
from alfi.trainers import VariationalTrainer
from .variational import TranscriptionLFM
from alfi.utilities.torch import q2
from torch.nn.functional import l1_loss
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from alfi.models import ExactLFM
from alfi.trainers import ExactTrainer
from collections import namedtuple

""" Experiment for finding mean absolute errors """
def get_datasets(data_dir='.'):
    from alfi.datasets import LFMDataset
    class DSet(LFMDataset):
        def __init__(self, disc, t_obs, m_obs, f_obs, lfm):
            self.num_outputs = 5
            self.m_observed = m_obs[:, :, ::disc]
            self.m_observed_highres = m_obs

            self.f_observed = f_obs[:, :, ::disc]
            self.f_observed_highres = f_obs

            self.t_observed = t_obs[::disc]
            self.t_observed_highres = t_obs
            self.lfm = lfm
            self.data = [(self.t_observed, self.m_observed[0, i]) for i in range(self.num_outputs)]
            self.gene_names = np.arange(self.num_outputs)

    datasets = list()
    for i in range(10):
        dataset_dict = torch.load(f'{data_dir}/experiments/dataset{i}.pt')
        lfm = namedtuple('lfm', ['basal_rate', 'sensitivity', 'decay_rate'])
        lfm = lfm(dataset_dict['basal'], dataset_dict['sensitivity'], dataset_dict['decay'])
        dataset = DSet(
            dataset_dict['high_disc'],
            dataset_dict['t_obs'],
            dataset_dict['m_obs'],
            dataset_dict['f_obs'], lfm)
        datasets.append(dataset)
    return datasets

def save_dataset(dataset, path):
    torch.save({
        'm_obs': dataset.m_observed_highres, 'high_disc': 10,
        't_obs': dataset.t_observed_highres,
        'f_obs': dataset.f_observed_highres,
        'lengthscale': dataset.lfm.lengthscale,
        'basal': dataset.lfm.basal_rate,
        'sensitivity': dataset.lfm.sensitivity,
        'decay': dataset.lfm.decay_rate
    }, path)


def get_params(lfm):
    if isinstance(lfm, ExactLFM):
        basal = lfm.mean_module.basal
        sensitivity = lfm.covar_module.sensitivity
        decay = lfm.covar_module.decay
    else:
        basal = lfm.basal_rate.squeeze()
        sensitivity = lfm.sensitivity.squeeze()
        decay = lfm.decay_rate.squeeze()
    return basal, sensitivity, decay

def param_l1(lfm, ground_truth):
    basal, sensitivity, decay = get_params(lfm)
    return (
        l1_loss(basal, torch.from_numpy(ground_truth[0])) +
        l1_loss(sensitivity, torch.from_numpy(ground_truth[1])) +
        l1_loss(decay, torch.from_numpy(ground_truth[2]))
    ).mean().item()

def train(lfm, trainer, m_targ, f_targ, epochs=200, **train_kwargs):
    f_maes = list()
    m_maes = list()
    f_q2s = list()
    m_q2s = list()
    i = 0
    for i in range(epochs // 10):
        try:
            trainer.train(epochs=10, report_interval=50, **train_kwargs)
        except:
            break
        m_pred = lfm.predict_m(t_predict, jitter=1e-3)
        f_pred = lfm.predict_f(t_predict, jitter=1e-3)
        m_mae = l1_loss(m_pred.mean, m_targ).mean().item()
        f_mae = l1_loss(f_pred.mean, f_targ).mean().item()
        f_q2 = q2(f_pred.mean, f_targ).mean().item()
        m_q2 = q2(m_pred.mean, m_targ).mean().item()
        f_maes.append(f_mae)
        m_maes.append(m_mae)
        f_q2s.append(f_q2)
        m_q2s.append(m_q2)
    if i < 10:
        print('except')
        raise Exception('too few epochs')
    f_maes = torch.tensor(f_maes)
    m_maes = torch.tensor(m_maes)
    min_index = (f_maes + m_maes).argmin(dim=0)
    f_mae = f_maes[min_index]
    m_mae = m_maes[min_index]

    f_q2s = torch.tensor(f_q2s)
    m_q2s = torch.tensor(m_q2s)
    min_index = (f_q2s + m_q2s).argmax(dim=0)
    f_q2 = f_q2s[min_index]
    m_q2 = m_q2s[min_index]

    return m_mae, f_mae, m_q2, f_q2


if __name__ == '__main__':
    generate = False
    if not generate:
        datasets = get_datasets()
    with open('experiments/mae.txt', 'w') as f:
        f.write(
            f'iter\tm_mae_exact\tm_mae_varia\tf_mae_exact\tf_mae_varia\tm_q2_exact'
            f'\tm_q2_varia\tf_q2_exact\tf_q2_varia\tparam_mae_exact\tparam_mae_varia\n')

        num_genes = 5
        num_tfs = 1
        config = VariationalConfiguration(
            num_samples=80,
            initial_conditions=False
        )
        maes = list()
        q2s = list()
        num_success = 0
        for i in range(20):
            if num_success >= 10:
                break
            print('Running ', i, num_success)
            if generate:
                exit()
                dataset = ToyTranscriptomicGenerator(
                    num_outputs=5,
                    num_latents=1,
                    num_times=12,
                    softplus=False,
                    dtype=torch.float32)
                dataset.generate_single(lengthscale=1.5)
            else:
                if i >= len(datasets):
                    break
                dataset = datasets[i]

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

            class ConstrainedTrainer(ExactTrainer):
                def after_epoch(self):
                    with torch.no_grad():
                        sens_fixed = dataset.lfm.sensitivity[0].squeeze().numpy()
                        dec_fixed = dataset.lfm.decay_rate[0].squeeze()
                        sens = self.lfm.covar_module.sensitivity
                        sens[0] = np.float64(sens_fixed)
                        deca = self.lfm.covar_module.decay
                        deca[0] = np.float64(dec_fixed)
                        self.lfm.covar_module.sensitivity = sens
                        self.lfm.covar_module.decay = deca
                    super().after_epoch()


            exact_trainer = ConstrainedTrainer(
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
                m_mae_exact, f_mae_exact,  m_q2_exact, f_q2_exact = train(exact_lfm, exact_trainer, m_targ, f_targ, epochs=700)
            except Exception as e:
                print('Exception:', e)
                # raise e
                continue

            #### Train variational
            num_inducing = 15  # (I x m x 1)
            inducing_points = torch.linspace(0, end_time, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
            gp_model = generate_multioutput_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=True))

            lfm = TranscriptionLFM(num_genes, gp_model, config)

            num_training = dataset.m_observed.shape[-1]

            variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.06)
            parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
            optimizers = [variational_optimizer, parameter_optimizer]

            class ConstrainedTrainer(VariationalTrainer):
                def after_epoch(self):
                    with torch.no_grad():
                        sens = dataset.lfm.sensitivity[0].squeeze()
                        dec = dataset.lfm.decay_rate[0].squeeze()
                        self.lfm.raw_sensitivity[0] = self.lfm.positivity.inverse_transform(sens)
                        self.lfm.raw_decay[0] = self.lfm.positivity.inverse_transform(dec)
                    super().after_epoch()


            trainer = ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=[
                'raw_basal',
                'raw_decay',
                'raw_sensitivity',
                'gp_model.covar_module.raw_lengthscale',
            ])

            lfm.train()
            try:
                m_mae_varia, f_mae_varia, m_q2_varia, f_q2_varia = train(lfm, trainer, m_targ, f_targ, epochs=700, step_size=5e-1)
            except Exception as e:
                print('exception', e)
                continue
            last_loss = trainer.losses[-1].sum()

            param_mae_exact = param_l1(exact_lfm, ground_truth)
            param_mae_varia = param_l1(lfm, ground_truth)
            print(m_mae_exact, m_mae_varia, f_mae_exact, f_mae_varia)
            maes.append(np.stack([m_mae_exact, m_mae_varia, f_mae_exact, f_mae_varia, param_mae_exact, param_mae_varia]))
            q2s.append(np.stack([m_q2_exact, m_q2_varia, f_q2_exact, f_q2_varia]))
            f.write(f'{i}\t{m_mae_exact}\t{m_mae_varia}\t{f_mae_exact}\t{f_mae_varia}\t{m_q2_exact}\t{m_q2_varia}'
                    f'\t{f_q2_exact}\t{f_q2_varia}\t{param_mae_exact}\t{param_mae_varia}\n')
            if generate:
                save_dataset(dataset, f'./experiments/dataset{num_success}.pt')
            num_success += 1
        maes = np.array(maes)
        q2s = np.array(q2s)
        np.save('./experiments/maes.npy', maes)
        np.save('./experiments/q2s.npy', q2s)
        print(maes)
        print(q2s)
        print(maes.shape)

        print('mae avg', maes.mean(axis=0), maes.std(axis=0))
        print('q2 avg', q2s.mean(axis=0), q2s.std(axis=0))
