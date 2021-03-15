from dataclasses import dataclass, field


@dataclass
class Configuration:
    num_outputs:            int
    num_latents:            int
    preprocessing_variance: object = None # data processing variances if present (e.g. mmgmos) otherwise None
    latent_data_present:    bool = False  # False for fully inferred latents
    delays:                 bool = False  # True if delay params used
    kernel:                 str = 'rbf'   # Kernel for latent function, rbf/mlp
    initial_conditions:     bool = True   # True if initial conditions for gene mRNA used
    translation:            bool = True   # True if the translation ODE is active
    kinetic_exponential:    bool = False  # True if kinetic params are exponentiated
    kernel_exponential:     bool = False  # True if kernel params are exponentiated


@dataclass
class VariationalConfiguration(Configuration):
    learn_inducing:         bool = False   # whether to learn the inducing point locations
    num_samples:            int = 20       # number of samples from the variational distribution
    kernel_scale:           bool = False   # whether to learn a scale parameter for the kernel


@dataclass
class MCMCConfiguration(Configuration):
    initial_step_sizes: dict = field(default_factory=dict)
    weights: bool = True  # True if weights used
