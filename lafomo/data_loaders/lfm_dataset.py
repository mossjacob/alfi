from torch.utils.data import Dataset
import abc


class LFMDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
