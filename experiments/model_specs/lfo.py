import torch

from torch.utils.data import DataLoader

from alfi.models import NeuralOperator
from alfi.trainers import NeuralOperatorTrainer


def build_dataset(dataset, ntest=50):
    batch_size = 50
    train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
    if ntest > 0:
        test_loader = DataLoader(dataset.test_data, batch_size=ntest, shuffle=True)
    return train_loader, test_loader


def build_lfo(dataset, modelparams, block_dim=2, **kwargs):
    train_loader, test_loader = dataset
    modes = 12
    width = 38
    in_channels = 3
    model = NeuralOperator(block_dim, in_channels, 2, modes, width, num_layers=4)
    learning_rate = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = NeuralOperatorTrainer(model, [optimizer], train_loader, test_loader)

    return model, trainer, None

def plot_lfo(dataset, lfm, trainer, plotter, filepath, params):
    pass
