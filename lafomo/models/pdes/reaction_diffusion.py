from fenics import *
from fenics_adjoint import *

import torch_fenics


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
