import torch
import math
import numpy as np
CUDA_AVAILABLE = False


def is_cuda():
    import torch
    return CUDA_AVAILABLE and torch.cuda.is_available()


def save(model, name):
    torch.save(model.state_dict(), f'./saved_models/{name}.pt')


def load(name, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(f'./saved_models/{name}.pt'))
    return model


def ceil(x):
    return int(math.ceil(x))


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


def get_image(data, intensity_index=2):
    """
    Returns an array compatible with plt.imshow
    Parameters:
        data: should be of shape (N, D) where N is the number of datapoints and D is the number of columns.
              First two columns are temporal and spatial dimensions
        intensity_index: the column index of the intensity of the image
    """
    ts = np.unique(data[:, 0])
    rows = list()
    diff = data[:-1, 0] - data[1:, 0]
    diff = int(np.ceil(np.abs(diff[np.nonzero(diff)][0])))+1
    for t in ts:
        row = data[data[:, 0] == t, intensity_index]
        rows.extend([row] * diff)
    return np.array(rows).T
