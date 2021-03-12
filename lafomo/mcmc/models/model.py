from lafomo.datasets import DataHolder
from lafomo.options import MCMCOptions
import abc

class MCMCLFM():

    def __init__(self, data: DataHolder, options: MCMCOptions):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_outputs = data.m_obs.shape[1]
        self.num_tfs = data.f_obs.shape[1]
        self.num_replicates = data.f_obs.shape[0]
        self.parameter_state = None

    @abc.abstractmethod
    def sample(self, T=2000, **kwargs):
        pass
