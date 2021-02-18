from dataclasses import dataclass

from reggae.gp import Options


@dataclass
class VariationalOptions(Options):
    learn_inducing: bool = False  # whether to learn the inducing point locations
    num_samples: int = 20              # number of samples from the variational distribution
    kernel_scale: bool = False          # whether to learn a scale parameter for the kernel
