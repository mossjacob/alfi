import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import abc

CUDA_AVAILABLE = False

class LFMDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index) -> T_co:
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


def is_cuda():
    import torch
    return CUDA_AVAILABLE and torch.cuda.is_available()


def save(model, name):
    torch.save(model.state_dict(), f'./saved_models/{name}.pt')


def load(name, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(f'./saved_models/{name}.pt'))
    return model


def softplus(value):
    return torch.log(1 + torch.exp(value))


def inv_softplus(value):
    return torch.log(torch.exp(value) - 1)


def cholesky_inverse(cholesky_factor, upper=False):
    """Courtesy of Alex Campbell"""
    batch_shape = list(cholesky_factor.shape[:-2])
    matrix_dim = list(cholesky_factor.shape[-2:])
    if batch_shape:
        flat_batch = torch.tensor(batch_shape).prod()
        inv_cholesky_factor = torch.stack([
            torch.cholesky_inverse(
                cholesky_factor.view(-1, *matrix_dim)[l], upper=upper)
            for l in range(flat_batch)])
        matrix = inv_cholesky_factor.view(*batch_shape, *matrix_dim)
    else:
        matrix = torch.cholesky_inverse(cholesky_factor, upper=upper)
    return matrix
