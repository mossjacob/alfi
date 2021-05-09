import torch
import math
import numpy as np
from torchcubicspline import(natural_cubic_spline_coeffs,
                             NaturalCubicSpline)
CUDA_AVAILABLE = True


def is_cuda():
    import torch
    return CUDA_AVAILABLE and torch.cuda.is_available()


def discretisation_length(N, d):
    """Returns the length of a linspace where there are N points with d intermediate (in-between) points."""
    return (N - 1) * (d + 1) + 1


def spline_interpolate_gradient(x: torch.Tensor, y: torch.Tensor, num_disc=9):
    """
    Returns x_interpolate, y_interpolate, y_grad, y_grad_2: the interpolated time, data and gradient
    """
    x_interpolate = torch.linspace(x.min(), x.max(), discretisation_length(x.shape[0], num_disc))
    coeffs = natural_cubic_spline_coeffs(x, y)
    spline = NaturalCubicSpline(coeffs)
    y_interpolate = spline.evaluate(x_interpolate)
    y_grad = spline.derivative(x_interpolate) #y_interpolate, denom, axis=1)
    y_grad_2 = spline.derivative(x_interpolate, order=2)
    return x_interpolate, y_interpolate, y_grad, y_grad_2


"""These metrics are translated from https://rdrr.io/cran/lineqGPR/man/errorMeasureRegress.html"""
def smse(y_test, f_mean):
    """Standardised mean square error (standardised by variance)"""
    return (y_test - f_mean).square() / y_test.var()


def q2(y_test, f_mean):
    y_mean = y_test.mean()
    return 1 - (y_test - f_mean).square().sum() / (y_test - y_mean).square().sum()


def cia(y_test, f_mean, f_var):
    return ((y_test >= (f_mean - 1.98 * f_var.sqrt())) &
            (y_test <= (f_mean + 1.98 * f_var.sqrt()))).double().mean()


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
    for t in ts:
        row = data[data[:, 0] == t, intensity_index]
        rows.extend([row])
    return np.array(rows).T

def discretise(time, num_discretised=40):
    """
    Calculates a discretised time grid with num_discretised points.
    Note: the return size is of size num_discretised + 1
    @param time: the time vector of size (2, N) where dim 0 is time, dim 1 is space.
    @param num_discretised: the number of points in the time grid
    @return: grid, time: discretised time grid as well as an updated time vector which is
    down or upsampled to include all points in the discretised time grid
    """
    t = np.unique(time)
    t.sort()
    t_range = t[-1] - t[0]
    dp = t_range / num_discretised
    print('t_sorted, dp', t, dp)
    return np.arange(t[0], t[-1] + dp, dp)


def get_mean_trace(trace):
    mean_trace = dict()
    for key in trace.keys():
        params = torch.stack(trace[key])
        for i in range(1, params.ndim):
            params = params.mean(-1)
        mean_trace[key] = params
    return mean_trace

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum('bix,iox->box', a, b)

def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum('bixy,ioxy->boxy', a, b)
