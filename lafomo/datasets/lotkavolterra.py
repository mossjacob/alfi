import torch
import numpy as np

from scipy.integrate import odeint
from . import LFMDataset
from tqdm import tqdm


class DeterministicLotkaVolterra(LFMDataset):
    """
    Dataset of Lotka-Voltera time series.
      Populations (u,v) evolve according to
        u' = \alpha u - \beta u v
        v' = \delta uv - \gamma v
      with the dataset sampled either with (u_0, v_0) fixed and (\alpha, \beta,
      \gamma, \delta) varied, or varying the initial populations for a fixed set
      of greeks.
    If initial values for (u,v) are provided then the greeks are sampled from
        (0.9,0.05,1.25,0.5) to (1.1,0.15,1.75,1.0)
    if values are provided for the greeks then (u_0 = v_0) is sampled from
        (0.5) to (2.0)
    if both are provided, defaults to initial population mode (greeks vary)
    ----------
    initial_u   : int
        fixed initial value for u
    initial_v   : int
        fixed initial value for v
    fixed_alpha : int
        fixed initial value for \alpha
    fixed_beta  : int
        fixed initial value for \beta
    fixed_gamma : int
        fixed initial value for \gamme
    fixed_delta : int
        fixed initial value for \delta
    end_time : float
        the final time (simulation runs from 0 to end_time)
    steps : int
        how many time steps to take from 0 to end_time
    """
    def __init__(self, initial_u=None, initial_v=None,
                 alpha=None, beta=None, gamma=None, delta=None, steps=150, end_time=15):

        if initial_u is None:
            self.mode = 'greek'
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
        else:
            self.mode = 'population'
            self.initial_u = initial_u
            self.initial_v = initial_v

        print(f'Lotka-Voltera is in {self.mode} mode.')

        self.steps = steps
        self.end_time = end_time

        # Generate data
        self.data = []
        print("Creating dataset...", flush=True)

        removed = 0
        times, states = self.generate_ts()
        #normalise times
        times = torch.FloatTensor(times) / 10

        states = torch.FloatTensor(states)
        if self.mode == 'population':
            states = states / 100
        #states = torch.cat((states, times), dim=-1)
        self.prey = states[:, 0]
        self.predator = states[:, 1]
        self.data.append((times[::5], states[::5, 1]))


    def generate_ts(self):
        if self.mode == 'population':
            X_0 = np.array([self.initial_u, self.initial_v])
            a = np.random.uniform(0.9, 1.1)
            b = np.random.uniform(0.05, 0.15)
            c = np.random.uniform(1.25, 1.75)
            d = np.random.uniform(0.5, 1.0)
        else:
            equal_pop = np.random.uniform(0.25,1.)
            X_0 = np.array([2*equal_pop,equal_pop])
            a, b, c, d = self.alpha, self.beta, self.gamma, self.delta
        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return np.array([ a*X[0] - b*X[0]*X[1],
                             -c*X[1] + d*X[0]*X[1]])

        t = np.linspace(0, self.end_time, self.steps)
        X = odeint(dX_dt, X_0, t)

        return t, X

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return 1
