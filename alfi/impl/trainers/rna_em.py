import torch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import DiagLazyTensor

from alfi.trainers import Trainer
from alfi.utilities.torch import ceil, is_cuda
from alfi.impl.odes import RNAVelocityLFM
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
    def __init__(self, lfm: RNAVelocityLFM, optimizers, dataset: LFMDataset, batch_size: int, **kwargs):
        super().__init__(lfm, optimizers, dataset, batch_size=batch_size, **kwargs)

    def e_step(self, y):
        # Given parameters, assign the timepoints
        num_outputs = self.lfm.num_outputs
        traj = self.lfm.current_trajectory
        u = traj[:num_outputs//2].unsqueeze(2)  # (num_genes, 100, 1)
        s = traj[num_outputs//2:].unsqueeze(2)  # (num_genes, 100, 1)
        u_y = y[:num_outputs//2]  # (num_genes, num_cells)
        s_y = y[num_outputs//2:]  # (num_genes, num_cells)

        batch_size = 500
        num_batches = ceil(y.shape[1] / batch_size)
        for batch in range(num_batches):  # batch over cells
            from_index = batch * batch_size
            to_index = (batch+1) * batch_size
            u_residual = u_y[:, from_index:to_index] - u.transpose(1, 2)
            s_residual = s_y[:, from_index:to_index] - s.transpose(1, 2)

            residual = u_residual.square() + s_residual.square()
            residual = residual.sum(dim=0).argmin(dim=1).type(torch.long)
            # print(residual.shape)
            # print(residual[:5])
            # print('done', batch)
            self.lfm.time_assignments_indices[from_index:to_index] = residual

    def random_assignment(self):
        self.lfm.time_assignments_indices = torch.randint(self.lfm.timepoint_choices.shape[0], torch.Size([self.lfm.num_cells]))

    def single_epoch(self, epoch=0, step_size=1e-1, **kwargs):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            [optim.zero_grad() for optim in self.optimizers]
            y = data.permute(0, 2, 1)  # (O, C, 1)
            y = y.cuda() if is_cuda() else y
            ### E-step ###
            # assign timepoints $t_i$ to each cell by minimising its distance to the trajectory
            # if epoch > 0:
            with torch.no_grad():
                self.e_step(y)
            # print('estep done')
            # else:
            #     self.random_assignment()

            ### M-step ###
            # TODO try to do it for only the time assignments
            # given timepoints, maximise for parameters
            t_sorted, inv_indices = torch.unique(self.lfm.time_assignments_indices, sorted=True, return_inverse=True)
            print('num t2:', t_sorted.shape)
            '''
            output = self.model(t_sorted, initial_value, rtol=rtol, atol=atol)
            output = output[:, inv_indices]
            # print(t_sorted, inv_indices.shape)
            '''
            # Get trajectory
            output = self.lfm(self.lfm.timepoint_choices, step_size=step_size)
            # print(output.shape)
            y = y.squeeze(-1)
            y *= self.lfm.nonzero_mask
            y_target = y.permute(1, 0)
            # print((output.mean - y_target).square().sum())
            # Calc loss and backprop gradients
            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target, mask=self.train_mask)
            total_loss = (-log_likelihood + kl_divergence)
            total_loss.backward()
            [optim.step() for optim in self.optimizers]

            epoch_loss += total_loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()
        return epoch_loss, (-epoch_ll, epoch_kl)
