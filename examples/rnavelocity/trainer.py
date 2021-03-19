import torch

from lafomo.variational.trainer import Trainer
from lafomo.utilities.torch import ceil, is_cuda


class EMTrainer(Trainer):
    """
    Expectation-Maximisation Trainer

    Parameters
    ----------
    model: .
    optimizer:
    dataset: Dataset where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainer should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self, model, optimizer: torch.optim.Optimizer, dataset, batch_size=1, give_output=False):
        super().__init__(model, optimizer, dataset, batch_size, give_output)
        # Initialise trajectory
        self.timepoint_choices = torch.linspace(0, 1, 100, requires_grad=False)
        initial_value = self.initial_value(None)
        self.previous_trajectory, _ = self.model(self.timepoint_choices, initial_value, rtol=1e-3, atol=1e-4)
        self.time_assignments_indices = torch.zeros_like(self.model.time_assignments, dtype=torch.long)

    def e_step(self, y):
        num_outputs = self.model.num_outputs
        # sorted_times, sort_indices = torch.sort(self.model.time_assignments, dim=0)
        # trajectory = self.model(sorted_times, self.initial_value(None), rtol=1e-2, atol=1e-3)

        # optimizer = torch.optim.LBFGS([model.time_assignments])
        u = self.previous_trajectory[:num_outputs//2]  # (num_genes, 100, 1)
        s = self.previous_trajectory[num_outputs//2:]  # (num_genes, 100, 1)
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
            self.model.time_assignments[from_index:to_index] = self.timepoint_choices[residual]
            self.time_assignments_indices[from_index:to_index] = residual

    def single_epoch(self, rtol, atol):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            y = data.permute(0, 2, 1) # (O, C, 1)
            y = y.cuda() if is_cuda() else y
            ### E-step ###
            # assign timepoints $t_i$ to each cell by minimising its distance to the trajectory
            self.e_step(y)
            print('estep done')

            ### M-step ###
            initial_value = self.initial_value(None)
            # TODO try to do it for only the time assignments
            t_sorted, inv_indices = torch.unique(self.model.time_assignments, sorted=True, return_inverse=True)
            print('num t:', t_sorted.shape)
            '''
            output = self.model(t_sorted, initial_value, rtol=rtol, atol=atol)
            output = output[:, inv_indices]
            # print(t_sorted, inv_indices.shape)
            '''
            y_mean, y_var = self.model(self.timepoint_choices, initial_value, rtol=rtol, atol=atol)

            self.previous_trajectory = y_mean
            y_mean = y_mean[:, self.time_assignments_indices]
            y_var = y_var[:, self.time_assignments_indices]
            print('ymean, yvar, traj', y_mean.shape, y_var.shape, self.previous_trajectory.shape)
            # output = torch.squeeze(output) # (num_genes, num_cells)
            # print(output.shape, y.shape)
            # Calc loss and backprop gradients

            mult = 1
            if self.num_epochs <= 10:
                mult = self.num_epochs/10

            ll, kl = self.model.elbo(y.squeeze(),
                                     y_mean.squeeze(),
                                     y_var.squeeze(),
                                     kl_mult=mult)
            total_loss = -ll# + kl
            total_loss.backward()
            self.optimizer.step()
            epoch_loss += total_loss.item()
            epoch_ll += ll.item()
            epoch_kl += kl.item()
        return epoch_loss, (-epoch_ll, epoch_kl)

    def after_epoch(self):
        with torch.no_grad():
            # TODO can we replace these with parameter transforms like we did with lengthscale
            self.model.transcription_rate.clamp_(0, 20)
            self.model.splicing_rate.clamp_(0, 20)
            self.model.decay_rate.clamp_(0, 20)
