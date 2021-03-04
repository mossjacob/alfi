from torch.utils.data import Dataset
import abc


class LFMDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index):
        return self.data[index]

    @property
    def num_outputs(self):
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, value):
        self._num_outputs = value

    @property
    def num_latents(self):
        return self._num_latents

    @num_latents.setter
    def num_latents(self, value):
        self._num_latents = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __len__(self):
        return len(self.data)
