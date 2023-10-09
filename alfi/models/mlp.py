import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, latent_dim=None, num_hidden_layers=3):
        super().__init__()
        latent_dim = 2 * ((input_dim + output_dim) // 2) if latent_dim is None else latent_dim
        middle_layers = list()
        for i in range(num_hidden_layers):
            middle_layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
            ])
        self.network = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            *middle_layers,
        )
        self.output_head = nn.Sequential(
            nn.Linear(latent_dim, output_dim)
        )
        self.latent_head = nn.Sequential(
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x, use_output_head=True):
        x = self.network(x)
        # if use_output_head:
        return self.output_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output_tensor = self(x)
        mse = F.mse_loss(output_tensor.squeeze(), y)
        self.log("train_loss", mse.item())
        return mse

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output_tensor = self(x)
        mse = F.mse_loss(output_tensor.squeeze(), y)
        self.log("val_loss", mse.item())
        return mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
