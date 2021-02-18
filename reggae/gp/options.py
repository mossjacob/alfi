from dataclasses import dataclass

@dataclass
class Options:
    preprocessing_variance: object = None # data processing variances if present (e.g. mmgmos) otherwise None
    tf_mrna_present:        bool = False  # False for inferred protein
    delays:                 bool = False  # True if delay params used
    kernel:                 str = 'rbf'   # Kernel for latent function, rbf/mlp
    initial_conditions:     bool = True   # True if initial conditions for gene mRNA used
    translation:            bool = True   # True if the translation mechanism is active
    kinetic_exponential:    bool = False  # True if kinetic params are exponentiated
    kernel_exponential:     bool = False  # True if kernel params are exponentiated
    