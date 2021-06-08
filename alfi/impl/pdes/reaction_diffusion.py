import torch
import torch_fenics

from fenics import *
from fenics_adjoint import *

from alfi.utilities.torch import spline_interpolate_gradient


class ReactionDiffusion(torch_fenics.FEniCSModule):
    def __init__(self, t_range: tuple, time_steps, mesh):
        """
        Parameters:
            t_range: tuple(low, high)
            dt: scalar float
            mesh_cells: Number of cells in the spatial mesh
        """
        super().__init__()
        self.time_steps = time_steps
        self.dt = ((t_range[1] - t_range[0]) / self.time_steps).item()
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, 'P', 1)

        # Create trial and test functions
        y = TrialFunction(self.V)
        self.v = TestFunction(self.V)

    def interpolated_gradient(self, tx, y_matrix, disc=1, plot=False):
        num_t = tx[0, :].unique().shape[0]
        num_x = tx[1, :].unique().shape[0]
        num_t_orig = y_matrix.shape[-2]
        num_x_orig = y_matrix.shape[-1]
        dy_t = list()
        for i in range(num_x_orig):
            t = tx[0][::num_x][::disc]
            y = y_matrix[:, i].unsqueeze(-1)
            t_interpolate, y_interpolate, y_grad, _ = \
                spline_interpolate_gradient(t, y)
            dy_t.append(y_grad)
        dy_t = torch.stack(dy_t)

        d2y_x = list()
        dy_x = list()
        for i in range(num_t_orig):
            t = tx[1][::disc][:num_x]
            y = y_matrix[i].unsqueeze(-1)
            t_interpolate, y_interpolate, y_grad, y_grad_2 = \
                spline_interpolate_gradient(t, y)
            d2y_x.append(y_grad_2)
            dy_x.append(y_grad)

        d2y_x = torch.stack(d2y_x)
        dy_x = torch.stack(dy_x)[..., ::10, 0].reshape(1, -1)
        d2y_x = d2y_x[..., ::10, 0].reshape(1, -1)
        dy_t = dy_t[..., ::10, 0].t().reshape(1, -1)

        def pde_func(y, u, sensitivity, decay, diffusion):
            # y (1, 1681) u (25, 1, 41, 41) s (25, 1)
            dy_t = (sensitivity * u.view(u.shape[0], -1) -
                    decay * y.view(1, -1) +
                    diffusion * d2y_x)
            return dy_t

        if plot:
            from alfi.plot import plot_spatiotemporal_data
            ts = tx[0, :].unique().numpy()
            xs = tx[1, :].unique().numpy()
            extent = [ts[0], ts[-1], xs[0], xs[-1]]
            axes = plot_spatiotemporal_data(
                [
                    y_matrix.view(num_t_orig, num_x_orig).t(),
                    dy_t.reshape(num_t_orig, num_x_orig).t(),
                    d2y_x.view(num_t_orig, num_x_orig).t(),
                ],
                extent, titles=['y', 'dy_t', 'd2y_x']
            )

        return pde_func, dy_t.t()

    def solve(self, y_prev, u, sensitivity, decay, diffusion):
        # Construct bilinear form (Arity = 2 (for both Trial and Test function))
        y = TrialFunction(self.V)
        self.a = (1 + self.dt * decay) * y * self.v * dx + self.dt * diffusion * inner(grad(y), grad(self.v)) * dx

        # Construct linear form
        L = (y_prev + self.dt * sensitivity * u) * self.v * dx

        # Construct boundary condition
        bc = DirichletBC(self.V, Constant(0), 'on_boundary')

        # Solve the equation
        y = Function(self.V)
        solve(self.a == L, y, bc)

        return y

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return Function(self.V), Function(self.V), \
               Constant(0), Constant(0), Constant(0)
