import torch
from torch import nn

from .model import OrdinaryLFM
from lafomo.datasets import LFMDataset


class MLPLFM(OrdinaryLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, **kwargs)
        h_dim = 20  # number of hidden units
        ode_layers = [nn.Linear(num_latents, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, num_outputs)]

        self.mlp = nn.Sequential(*ode_layers)

    def odefunc(self, t, h):
        """
        h shape (num_outputs, 1)
        """
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample()
        if self.extra_points > 0:
            f = f[:, self.extra_points]  # get the midpoint

        y = self.mlp(f)
        y = torch.unsqueeze(y, 1)
        return y


class ConvLFM(OrdinaryLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, dtype=torch.float32, **kwargs)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        n_filt = 16
        img_width = 28
        self.img_width_flat = 28 * 28
        num_hidden = 20
        encoder_layers = [nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2, 2)),
                          nn.BatchNorm2d(16),
                          nn.ReLU(),
                          nn.Conv2d(n_filt, n_filt * 2, kernel_size=5, stride=2, padding=(2, 2)),
                          nn.BatchNorm2d(32),
                          nn.ReLU(),
                          nn.Conv2d(n_filt * 2, n_filt * 4, kernel_size=5, stride=2, padding=(2, 2)),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),
                          nn.Conv2d(n_filt * 4, n_filt * 8, kernel_size=5, stride=2, padding=(2, 2)),
                          nn.Flatten(),
                          nn.Linear(512, num_hidden)
                          ]

        feat = 3 if img_width == 28 else 4
        print(img_width)
        padding1 = (0, 0) if img_width == 28 else (2, 2)
        padding2 = (0, 0) if img_width == 28 else (1, 1)
        stride = 1 if img_width == 28 else 2
        k_size = 3 if img_width == 28 else 5
        decoder_layers = [nn.ConvTranspose2d(32, 128, kernel_size=k_size, stride=stride, padding=padding1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=padding2),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),
                          nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
                          nn.BatchNorm2d(32),
                          nn.ReLU(),
                          nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=(2, 2))
                          ]
        self.decoder_fc = nn.Linear(num_hidden, feat * feat * 32)
        self.encoder = nn.Sequential(*encoder_layers).float()
        self.decoder = nn.Sequential(*decoder_layers).float()

        h_dim = 50
        ode_layers = [nn.Linear(num_hidden + num_latents, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, num_hidden)]

        self.mlp = nn.Sequential(*ode_layers)


    def initial_state(self, y):
        """
        Called before the ODE is solved in order to get the initial state
        """
        # print('y0', y.shape)
        h0 = self.encoder(y.view(1, 1, 28, 28))  # (B, C, W, W)
        # print('h0', h0.shape)
        return h0

    def decode(self, h_out):
        """h_out shape (B, T, num_hidden)"""
        # print('hout', h_out.shape)
        # TODO: we temporarily make the T the batch dimension here...
        h_out = torch.squeeze(h_out, 0)
        # TODO: end
        h_out = self.decoder_fc(h_out)
        h_out = h_out.view(h_out.shape[0], 32, 3, 3)
        y = self.decoder(h_out)
        y = y.view(y.shape[0], y.shape[1], self.img_width_flat)
        # print('yout', y.shape)
        return y

    def odefunc(self, t, h):
        """
        h shape (num_outputs, 1)
        """
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample()
        if self.extra_points > 0:
            f = f[:, self.extra_points]  # get the midpoint

        f = torch.unsqueeze(f, 0)

        input = torch.cat([h, f], dim=1)
        h_evolved = self.mlp(input)
        return h_evolved

    def log_likelihood(self, y, h):
        log_lik = self.loss(h, y)
        log_lik = - torch.sum(log_lik)  # minus since it's already a loss
        return log_lik
