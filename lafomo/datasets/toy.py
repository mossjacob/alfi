import torch

from torch.nn import Parameter
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO
import numpy as np

from matplotlib import pyplot as plt

from .datasets import TranscriptomicTimeSeries
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import softplus



class ToyTimeSeries(TranscriptomicTimeSeries):
    """
    This dataset stochastically generates a toy transcriptional regulation dataset.
    """

    def __init__(self, num_outputs=30, num_latents=3, num_times=10):
        super().__init__()
        from lafomo.models import OrdinaryLFM, MultiOutputGP
        from lafomo.plot import Plotter
        class ToyLFM(OrdinaryLFM):
            """
            This LFM is to generate toy data.
            """

            def __init__(self, num_outputs, gp_model, config: VariationalConfiguration):
                super().__init__(num_outputs, gp_model, config)
                num_latents = gp_model.variational_strategy.num_tasks
                self.decay_rate = Parameter(
                    0.2 + 2 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                self.basal_rate = Parameter(
                    0.1 + 0.3 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                self.sensitivity = Parameter(2 + 5 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                weight = 0.5 + 1 * torch.randn(torch.Size([self.num_outputs, num_latents]), dtype=torch.float32)
                weight[torch.randperm(self.num_outputs)[:15], torch.randint(num_latents, [15])] = 0
                self.weight = Parameter(weight)
                self.weight_bias = Parameter(torch.randn(torch.Size([self.num_outputs, 1]), dtype=torch.float32))

            def initial_state(self):
                return self.basal_rate / self.decay_rate

            def odefunc(self, t, h):
                """h is of shape (num_samples, num_outputs, 1)"""
                self.nfe += 1
                decay = self.decay_rate * h
                f = self.f[:, :, self.t_index].unsqueeze(2)
                h = self.basal_rate + f - decay
                if t > self.last_t:
                    self.t_index += 1
                self.last_t = t
                return h

            def G(self, f):
                f = softplus(f)
                interactions = torch.matmul(self.weight, torch.log(f + 1e-100)) + self.weight_bias
                f = torch.sigmoid(interactions)  # TF Activation Function (sigmoid)
                return f

        self.num_outputs = num_outputs
        self.num_latents = num_latents
        config = VariationalConfiguration(
            num_samples=70,
            kernel_scale=False,
            initial_conditions=False # TODO
        )
        prediction_points = num_times * 10
        num_inducing = 10  # (I x m x 1)
        inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_latents, 1).view(num_latents, num_inducing, 1)
        t_predict = torch.linspace(0, 12, prediction_points, dtype=torch.float32)

        gp_model = MultiOutputGP(inducing_points, num_latents, initial_lengthscale=2, natural=False)
        self.train_gp(gp_model, t_predict)
        with torch.no_grad():
            lfm = ToyLFM(num_outputs, gp_model, config)
            plotter = Plotter(lfm, np.arange(num_outputs))
            q_m = plotter.plot_gp(lfm.predict_m(t_predict), t_predict)
            self.t_observed = t_predict[::10]
            self.m_observed = q_m.mean[::10].unsqueeze(0).permute(0, 2, 1)
            print(self.m_observed.shape, self.f_observed.shape)
            self.data = [(self.t_observed, self.m_observed[0, i]) for i in range(num_outputs)]

    def train_gp(self, gp_model, t_predict):
        q_f = gp_model(t_predict)
        samples = q_f.sample(torch.Size([1]))

        fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
        for i in range(samples.shape[0]):
            axes[0].plot(softplus(samples[i][:, 0]), color='blue')
            axes[0].plot(softplus(samples[i][:, 1]), color='red')
            axes[0].plot(softplus(samples[i][:, 2]), color='yellow')
        train_y = samples[0]
        self.f_observed = train_y[::10].unsqueeze(0).permute(0, 2, 1)

        likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_latents)

        mll = VariationalELBO(likelihood, gp_model, num_data=train_y.size(0))
        optimizer = torch.optim.Adam([
            {'params': gp_model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.1)

        #train gp_model
        for i in range(100):
            optimizer.zero_grad()
            output = gp_model(t_predict)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        q_f = gp_model(t_predict)
        samples = q_f.sample(torch.Size([1]))

        for i in range(samples.shape[0]):
            axes[1].plot(softplus(samples[i][:, 0]), color='blue')
            axes[1].plot(softplus(samples[i][:, 1]), color='red')
            axes[1].plot(softplus(samples[i][:, 2]), color='yellow')
