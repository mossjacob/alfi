import torch
import numpy as np

from sklearn.cluster import KMeans

from alfi.trainers import Trainer
from alfi.utilities.torch import ceil, is_cuda, spline_interpolate_gradient, savgol_filter_gradient
from alfi.impl.odes import RNAVelocityLFM
from alfi.datasets import LFMDataset
from alfi.models import TrainMode


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
        cells = next(iter(self.data_loader))
        X = cells.squeeze().t()
        self.num_clusters = 8
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(X)
        print(self.kmeans.labels_.shape)
        print(self.kmeans.predict(X).shape)
        print(self.kmeans.cluster_centers_.shape)

    def e_step(self, y, add_penalty=False):
        # Given parameters, assign the timepoints
        num_outputs = self.lfm.num_outputs
        traj = self.lfm.current_trajectory
        u = traj[:num_outputs//2].unsqueeze(2)  # (num_genes, 100, 1)
        s = traj[num_outputs//2:].unsqueeze(2)  # (num_genes, 100, 1)
        u_y = y[:num_outputs//2]  # (num_genes, num_cells)
        s_y = y[num_outputs//2:]  # (num_genes, num_cells)

        time_assignments = self.lfm.time_assignments_indices
        cluster_mean_times = torch.tensor(  # the mean times per cluster (num_clusters)
            [time_assignments.type(torch.float)[self.kmeans.labels_ == c].mean() for c in range(self.num_clusters)])
        print(cluster_mean_times)
        mu_cells = cluster_mean_times[self.kmeans.labels_]  # mean time for the cluster each cell belongs to (num_cells)
        batch_size = 500
        num_batches = ceil(y.shape[1] / batch_size)
        for batch in range(num_batches):  # batch over cells. we assign times per cell
            from_index = batch * batch_size
            to_index = (batch+1) * batch_size
            u_residual = u_y[:, from_index:to_index] - u.transpose(1, 2)
            s_residual = s_y[:, from_index:to_index] - s.transpose(1, 2)

            # residual is now the indices into the time vector (500,)
            # add penalty
            # diff in cell's cluster (small desired) - avg diff to other clusters (large desired)
            mu_cells_batch = mu_cells[from_index:to_index]  # (num_cells, 1) cluster means for all cells

            within_cluster_diff = (self.lfm.timepoint_choices.unsqueeze(0) - mu_cells_batch.unsqueeze(1)).square()  # (num_cells, 200)
            # other_cluster_diff = torch.empty((batch_size, self.lfm.timepoint_choices.shape[0]))
            # for cluster in range(self.num_clusters):
            #     cluster_mask = self.kmeans.labels_ == cluster
            #     cluster_mean_time = cluster_mean_times[cluster]
            #     # get difference between cluster mean for cell and the timepoint choices
            #     diff = self.lfm.timepoint_choices.unsqueeze(0) - mu_cells_batch.unsqueeze(1)  # (num_cells, 200)
            #     # mask out those differences to the same cluster
            #     diff[cluster_mask] *= 0
            #     other_cluster_diff += diff.square()
            print(mu_cells_batch.shape, '-', time_assignments[from_index:to_index].shape)
            residual = u_residual.square() + s_residual.square()
            residual = residual.sum(dim=0)
            if add_penalty:
                residual += within_cluster_diff #- other_cluster_diff # (batch, 200)
            print('resid shape', residual.shape)
            residual = residual.argmin(dim=1).type(torch.long)  # sum over genes (batch) TODO: average??

            self.lfm.time_assignments_indices[from_index:to_index] = residual.squeeze()

    def random_assignment(self):
        self.lfm.time_assignments_indices = torch.randint(self.lfm.timepoint_choices.shape[0], torch.Size([self.lfm.num_cells]))

    def get_interpolated_data(self, cells):
        num_outputs = self.lfm.num_outputs
        u_y = cells[:num_outputs // 2]  # (num_genes, num_cells)
        s_y = cells[num_outputs // 2:]  # (num_genes, num_cells)

        # First step: sort the cells in the current assignment order
        sorted_t, sorted_ind = self.lfm.time_assignments_indices.sort()
        sorted_t = self.lfm.timepoint_choices[sorted_t]
        u_y = u_y[:, sorted_ind].squeeze(-1)
        s_y = s_y[:, sorted_ind].squeeze(-1)

        # Second step: bucket the cells based on their time assigment
        u_y_bucketed = list()
        s_y_bucketed = list()
        for t in self.lfm.timepoint_choices:
            a = sorted_t == t
            u_y_bucketed.append(u_y[:, a].mean(-1))
            s_y_bucketed.append(s_y[:, a].mean(-1))
        u_y_bucketed = torch.stack(u_y_bucketed, dim=1)
        s_y_bucketed = torch.stack(s_y_bucketed, dim=1)

        # Third step: interpolate the buckets to ensure regular time spacing
        t_interpolate, u_y_interp, _, _ = spline_interpolate_gradient(
            self.lfm.timepoint_choices,
            u_y_bucketed.unsqueeze(-1),
            num_disc=2)
        _, s_y_interp, _, _ = spline_interpolate_gradient(
            self.lfm.timepoint_choices,
            s_y_bucketed.unsqueeze(-1),
            num_disc=2)

        # Fourth step: denoise and get the first derivative
        u_y_denoised, du = savgol_filter_gradient(t_interpolate, u_y_interp.squeeze(-1))
        s_y_denoised, ds = savgol_filter_gradient(t_interpolate, s_y_interp.squeeze(-1))
        du, ds, u_y_denoised, s_y_denoised = [
            torch.tensor(x) for x in [du, ds, u_y_denoised, s_y_denoised]]
        if du.ndim < 2:
            du, ds, u_y_denoised, s_y_denoised = [
                x.unsqueeze(0) for x in [du, ds, u_y_denoised, s_y_denoised]]
        du, ds, u_y_denoised, s_y_denoised = [
            x[:, ::3] for x in [du, ds, u_y_denoised, s_y_denoised]]
        data_interpolated_gradient = torch.cat([du, ds], dim=0).t()
        data_interpolated = torch.cat([u_y_denoised, s_y_denoised], dim=0)
        t_interpolate = t_interpolate[::3]
        return t_interpolate, data_interpolated, data_interpolated_gradient

    def single_epoch(self, epoch=0, step_size=1e-1, warmup=-1, **kwargs):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            [optim.zero_grad() for optim in self.optimizers]
            y = data.permute(0, 2, 1)  # (O, C, 1)
            y = y.cuda() if is_cuda() else y
            cells = y[:-1] if self.lfm.config.latent_data_present else y
            if self.lfm.train_mode == TrainMode.FILTER:
                t_interpolated, data_interpolated, _ = self.get_interpolated_data(cells)
                y_target = data_interpolated.t()
                x = self.lfm.timepoint_choices
            elif self.lfm.train_mode == TrainMode.GRADIENT_MATCH:
                t_interpolated, data_interpolated, data_interpolated_gradient = self.get_interpolated_data(cells)
                y_target = data_interpolated_gradient
                x = (t_interpolated, data_interpolated)
            else:
                y = y.squeeze(-1)
                y *= self.lfm.nonzero_mask
                y_target = y.permute(1, 0)
                x = self.lfm.timepoint_choices

            ### E-step ###
            # assign timepoints $t_i$ to each cell by minimising its distance to the trajectory
            # if epoch > 0:
            e_step = 5 if self.lfm.train_mode == TrainMode.GRADIENT_MATCH else 1
            if (epoch % e_step) == 0:
                with torch.no_grad():
                    if not (self.lfm.train_mode == TrainMode.NORMAL):
                        # If pretraining, then call the LFM with normal mode
                        mode = self.lfm.train_mode
                        self.lfm.set_mode(TrainMode.NORMAL)
                        self.lfm(self.lfm.timepoint_choices, step_size=step_size)
                        self.e_step(cells)
                        self.lfm.set_mode(mode)
                    else:
                        self.e_step(cells)
            # print('estep done')
            # else:
            #     self.random_assignment()

            ### M-step: given timepoints, maximise for parameters ###

            # TODO try to do it for only the time assignments
            t_sorted, inv_indices = torch.unique(self.lfm.time_assignments_indices, sorted=True, return_inverse=True)
            print('num t2:', t_sorted.shape)
            '''
            output = self.model(t_sorted, initial_value, rtol=rtol, atol=atol)
            output = output[:, inv_indices]
            # print(t_sorted, inv_indices.shape)
            '''
            # Get trajectory
            output = self.lfm(x, step_size=step_size)

            # Calc loss and backprop gradients
            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target, mask=self.train_mask)
            total_loss = (-log_likelihood + kl_divergence)
            total_loss.backward()

            if epoch < warmup:
                self.optimizers[1].step()
            else:
                [optim.step() for optim in self.optimizers]

            epoch_loss += total_loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()
        return epoch_loss, (-epoch_ll, epoch_kl)
