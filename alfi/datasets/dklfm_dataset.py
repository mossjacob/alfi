import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


class DeepKernelLFMDataset(Dataset):
    """
    VelocityOperatorDataset
    This holds
    """
    n_blocks = None
    """number of timepoint blocks corresponding to a different output"""
    train_indices = None
    """Indices of the training timepoints"""
    train_instance_indices = None
    """Indices of the training tasks"""
    x_cond = None
    """Timepoints corresponding to the conditioning data"""
    x_cond_blocks = None
    """Timepoints blocks (one block per output)"""
    y_cond = None
    """Conditioning outputs"""
    input_dim = 1
    """Dimensions of input. Default is 1. When 2, will plot input as images with output as colour."""

    def __init__(self, x, y, f=None, train_indices=None, n_train=0.6, n_training_instances=0.5, scale_x_max=False, n_test_instances=1.):
        """
        Args:
            x: timepoints or inputs (num_inputs, dimensions)
            y: shape (num_tasks, n_blocks, num_times)
            n_train: number of training timepoints
            n_training_instances: number of training instances
        """
        self.n_task = y.shape[0]
        self.n_blocks = y.shape[1]
        self.timepoints = x
        self.y = y
        self.f = f
        self.input_dim = x.shape[-1] if x.ndim > 1 else 1
        if n_training_instances < 1.:
            n_training_instances = int(n_training_instances * self.n_task)
        if train_indices is None:
            n_times = y.shape[2]
            if n_train < 1.:
                n_train = int(n_train * n_times)
            train_indices = np.random.permutation(np.arange(n_times))[:n_train]
            train_indices = np.sort(train_indices)
        self.train_indices = train_indices
        self.timepoints_cond = x[self.train_indices]

        self.x_cond = self.timepoints_cond.view(self.timepoints_cond.shape[0], -1).unsqueeze(0).repeat(self.n_task, 1, 1).type(torch.float64)
        if scale_x_max:
            self.x_cond /= self.x_cond.max()
        self.x_cond_blocks = self.x_cond.repeat(1, self.n_blocks, 1)
        shape = (self.n_task, -1)
        self.y_cond = y[..., self.train_indices].reshape(shape)  # (n_task, 2, T) -> (n_task, 2 * T)
        self.f_cond = f if f is None else f[..., self.train_indices].reshape(shape)

        n_test_instances = int(n_test_instances * self.x_cond.shape[0]) if isinstance(n_test_instances, float) else n_test_instances
        self.train_instance_indices = np.arange(n_training_instances)
        self.test_instance_indices = np.arange(n_training_instances, n_training_instances + n_test_instances)

        self.train_x_cond_blocks = self.x_cond_blocks[self.train_instance_indices]
        self.train_x_cond = self.x_cond[self.train_instance_indices]
        self.train_y_cond = self.y_cond[self.train_instance_indices]
        self.train_f_cond = f if f is None else self.f_cond[self.train_instance_indices]

    def __len__(self):
        return self.train_x_cond.shape[0]

    def __getitem__(self, i):
        return (
            self.train_x_cond_blocks[i],
            self.train_x_cond[i],
            self.train_y_cond[i],
            self.train_f_cond[i]
        )

    def validation(self):
        return TensorDataset(
            self.x_cond_blocks[self.test_instance_indices],#self.train_x_cond_blocks,
            self.x_cond[self.test_instance_indices], #self.train_x_cond,
            self.y_cond[self.test_instance_indices], #self.train_y_cond,
            self.f_cond[self.test_instance_indices] #self.train_f_cond
        )