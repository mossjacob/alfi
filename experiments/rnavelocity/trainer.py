import torch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

from alfi.trainers import Trainer
from alfi.utilities.torch import ceil, is_cuda
from alfi.models import LFM
from alfi.datasets import LFMDataset


class EMTrainer(Trainer):
    """
    Expectation-Maximisation Trainer

    Parameters
    ----------
    de_model: .
    optimizers:
    dataset: Dataset where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainer should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self, lfm: LFM, optimizers: torch.optim.Optimizer, dataset: LFMDataset, batch_size: int):
        super().__init__(lfm, optimizers, dataset, batch_size=batch_size)
        # Initialise trajectory
        self.timepoint_choices = torch.linspace(0, 1, 100, requires_grad=False)
        initial_value = self.initial_value(None)
        self.previous_trajectory = self.lfm(self.timepoint_choices, step_size=1e-2)
        self.time_assignments_indices = torch.zeros_like(self.lfm.time_assignments, dtype=torch.long)

    def e_step(self, y):
        num_outputs = self.lfm.num_outputs
        # sorted_times, sort_indices = torch.sort(self.model.time_assignments, dim=0)
        # trajectory = self.model(sorted_times, self.initial_value(None), rtol=1e-2, atol=1e-3)

        # optimizer = torch.optim.LBFGS([model.time_assignments])
        traj = self.previous_trajectory.mean.transpose(0, 1)
        u = traj[:num_outputs//2].unsqueeze(2)  # (num_genes, 100, 1)
        s = traj[num_outputs//2:].unsqueeze(2)  # (num_genes, 100, 1)
        u_y = y[:num_outputs//2]  # (num_genes, num_cells)
        s_y = y[num_outputs//2:]  # (num_genes, num_cells)

        batch_size = 500
        num_batches = ceil(y.shape[1] / batch_size)
        for batch in range(num_batches):
            from_index = batch * batch_size
            to_index = (batch+1) * batch_size
            u_residual = u_y[:, from_index:to_index] - u.transpose(1, 2)
            s_residual = s_y[:, from_index:to_index] - s.transpose(1, 2)

            residual = u_residual.square() + s_residual.square()
            residual = residual.sum(dim=0).argmin(dim=1).type(torch.long)
            # print(residual.shape)
            # print(residual[:5])
            # print('done', batch)
            self.lfm.time_assignments[from_index:to_index] = self.timepoint_choices[residual]
            self.time_assignments_indices[from_index:to_index] = residual

    def single_epoch(self, step_size=1e-1):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            [optim.zero_grad() for optim in self.optimizers]
            y = data.permute(0, 2, 1)  # (O, C, 1)
            y = y.cuda() if is_cuda() else y
            ### E-step ###
            # assign timepoints $t_i$ to each cell by minimising its distance to the trajectory
            self.e_step(y)
            print('estep done')

            ### M-step ###
            # TODO try to do it for only the time assignments
            t_sorted, inv_indices = torch.unique(self.lfm.time_assignments, sorted=True, return_inverse=True)
            print('num t:', t_sorted.shape)
            '''
            output = self.model(t_sorted, initial_value, rtol=rtol, atol=atol)
            output = output[:, inv_indices]
            # print(t_sorted, inv_indices.shape)
            '''
            output = self.lfm(self.timepoint_choices, step_size=step_size)
            self.previous_trajectory = output

            f_mean = output.mean.transpose(0, 1)[:, self.time_assignments_indices]
            f_var = output.variance.transpose(0, 1)[:, self.time_assignments_indices]
            print(f_mean.shape, f_var.shape)
            f_covar = torch.diag_embed(f_var)
            batch_mvn = MultivariateNormal(f_mean, f_covar)
            output = MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

            # Calc loss and backprop gradients
            y_target = y.squeeze(-1).permute(1, 0)
            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target)
            total_loss = -log_likelihood + kl_divergence
            total_loss.backward()
            [optim.step() for optim in self.optimizers]

            epoch_loss += total_loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()
        return epoch_loss, (-epoch_ll, epoch_kl)

    def after_epoch(self):
        with torch.no_grad():
            # TODO can we replace these with parameter transforms like we did with lengthscale
            self.lfm.transcription_rate.clamp_(0, 20)
            self.lfm.splicing_rate.clamp_(0, 20)
            self.lfm.decay_rate.clamp_(0, 20)
