from torch.utils.data import Dataset
import abc


class LFMDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index):
        return self.data[index]

    @property
    def num_outputs(self):
        """The number of LFM outputs."""
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, value):
        self._num_outputs = value

    @property
    def data(self):
        """
        List of data points, each a tuple(a, b).
        For time-series, a and b are 1-D.
        For spatiotemporal series, a is (2, T) corresponding to a row for time and space, and b is 1-D.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __len__(self):
        return len(self.data)
