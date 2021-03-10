from lafomo.datasets import DataHolder
from lafomo.options import MCMCOptions


class MCMCLFM():
    """
    Likelihood of the form:
    N(m(t), s(t))
    where m(t) = b/d + (a - b/d) exp(-dt) + s int^t_0 G(p(u); w) exp(-d(t-u)) du
    """

    def __init__(self, data: DataHolder, options: MCMCOptions):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[1]
        self.num_tfs = data.f_obs.shape[1]
        self.num_replicates = data.f_obs.shape[0]
