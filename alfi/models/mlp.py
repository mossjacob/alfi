from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels, latent_dims, num_hidden_layers=3):
        super().__init__()
        mid_channels = 2 * ((in_channels + latent_dims) // 2)
        middle_layers = list()
        for i in range(num_hidden_layers):
            middle_layers.extend([
                nn.Linear(mid_channels, mid_channels),
                nn.ReLU(),
            ])
        self.network = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            *middle_layers,
        )
        self.output_head = nn.Sequential(
            nn.Linear(mid_channels, latent_dims)
        )
        self.latent_head = nn.Sequential(
            nn.Linear(mid_channels, latent_dims)
        )

    def forward(self, x, use_output_head=True):
        x = self.network(x)
        # if use_output_head:
        return self.output_head(x)
