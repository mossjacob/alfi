from dataclasses import dataclass, field

@dataclass
class Options:
    preprocessing_variance: bool = True  # True if data processing variances present (e.g. mmgmos)
    tf_mrna_present:        bool = True  # False for inferred protein
    delays:                 bool = False # True if delay params used
    kernel:                 str = 'rbf'  # Kernel for latent function, rbf/mlp
    joint_latent:           bool = True  # Whether to sample the latents jointly with hyperparams
    initial_step_sizes:     dict = field(default_factory=dict)
    weights:                bool = True  # True if weights used
    initial_conditions:     bool = True  # True if initial conditions for gene mRNA used
    translation:            bool = True  # True if the translation mechanism is active
    kinetic_exponential:    bool = False # True if kinetic params are exponentiated
    kernel_exponential:     bool = False # True if kernel params are exponentiated
    