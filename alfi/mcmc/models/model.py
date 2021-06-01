from alfi.datasets import DataHolder
from alfi.configuration import MCMCConfiguration
import abc

class MCMCLFM():

    def __init__(self, data: DataHolder, options: MCMCConfiguration):
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
