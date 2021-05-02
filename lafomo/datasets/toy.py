import torch
from pathlib import Path
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

    def __init__(self, num_outputs=30, num_latents=3, num_times=10, params=None, plot=True):
        super().__init__()
        self.num_disc = 9
        self.plot = plot
        from lafomo.models import OrdinaryLFM, generate_multioutput_rbf_gp

        class ToyLFM(OrdinaryLFM):
            """
            This LFM is to generate toy data.
            """

            def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
                super().__init__(num_outputs, gp_model, config, **kwargs)
                num_latents = gp_model.variational_strategy.num_tasks
                self.decay_rate = Parameter(params[2] if params is not None else
                                            0.2 + 2 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                self.basal_rate = Parameter(params[0] if params is not None else
                                            0.1 + 0.3 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                self.sensitivity = Parameter(params[1] if params is not None else
                                             2 + 5 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float32))
                weight = 0.5 + 1 * torch.randn(torch.Size([self.num_outputs, num_latents]), dtype=torch.float32)
                num_remove = num_outputs // 2
                weight[torch.randperm(self.num_outputs)[:num_remove], torch.randint(num_latents, [num_remove])] = 0
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
                if f.shape[1] > 1:
                    interactions = torch.matmul(self.weight, torch.log(f + 1e-100)) + self.weight_bias
                    f = torch.sigmoid(interactions)  # TF Activation Function (sigmoid)
                return f

        self.num_outputs = num_outputs
        self.num_latents = num_latents
        config = VariationalConfiguration(
            num_samples=70,
            initial_conditions=False # TODO
        )

        def calc(N, d):
            return (N - 1) * (d + 1) + 1

        num_inducing = 10  # (I x m x 1)
        inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_latents, 1).view(num_latents, num_inducing, 1)
        t_predict = torch.linspace(0, 12, calc(num_times, self.num_disc), dtype=torch.float32)

        gp_model = generate_multioutput_rbf_gp(num_latents, inducing_points,
                                               initial_lengthscale=2,
                                               gp_kwargs=dict(natural=False))
        self.train_gp(gp_model, t_predict)
        with torch.no_grad():
            self.lfm = ToyLFM(num_outputs, gp_model, config)
            q_m = self.lfm.predict_m(t_predict)
            self.t_observed_highres = t_predict
            self.t_observed = t_predict[::self.num_disc]
            self.m_observed_highres = q_m.mean.unsqueeze(0).permute(0, 2, 1)
            self.m_observed = q_m.mean[::self.num_disc].unsqueeze(0).permute(0, 2, 1)
            self.data = [(self.t_observed, self.m_observed[0, i]) for i in range(num_outputs)]

    def train_gp(self, gp_model, t_predict):
        q_f = gp_model(t_predict)
        samples = q_f.sample(torch.Size([1]))  # sample from the prior
        if self.plot:
            fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
            for i in range(samples.shape[0]):
                for l in range(self.num_latents):
                    axes[0].plot(softplus(samples[i][:, l]))
        train_y = samples[0]
        self.f_observed = train_y[::self.num_disc].unsqueeze(0).permute(0, 2, 1)

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
        if self.plot:
            for i in range(samples.shape[0]):
                for l in range(self.num_latents):
                    axes[1].plot(softplus(samples[i][:, l]))


class ToyTranscriptomics():
    def __init__(self, data_dir='../data'):
        data = torch.load(Path(data_dir) / 'toy_transcriptomics.pt')
        train = data['train_x']

        train = [(x_train[i], y_train[i], params[i].type(torch.float)) for i in range(ntrain)]
        test = [(x_test[i], y_test[i], params[i].type(torch.float)) for i in range(ntest)]

        self.train_data = train
        self.test_data = test


class TranscriptomicGenerator:

    def __init__(self):
        basal_rate = 0.1 + 0.3 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)
        sensitivity = 2 + 5 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)
        decay_rate = 0.2 + 2 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)

    def generate(self, ntrain, ntest, data_dir):
        num_outputs = 10
        datasets = list()
        for i in range(ntrain + ntest):
            basal_rate = 0.1 + 0.3 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)
            sensitivity = 2 + 5 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)
            decay_rate = 0.2 + 2 * torch.rand(torch.Size([num_outputs, 1]), dtype=torch.float32)

            dataset = ToyTimeSeries(num_outputs, 1, 10, params=[basal_rate, sensitivity, decay_rate], plot=False)
            datasets.append(dataset)
        x_train = torch.cat([dataset.m_observed for dataset in datasets[:ntrain]]).permute(0, 2, 1)
        x_test = torch.cat([dataset.m_observed for dataset in datasets[ntrain:]]).permute(0, 2, 1)
        grid = datasets[0].t_observed.reshape(1, -1, 1).repeat(ntrain, 1, 1) # (1, 32, 32, 40, 1)
        grid_test = datasets[0].t_observed.reshape(1, -1, 1).repeat(ntest, 1, 1) # (1, 32, 32, 40, 1)

        x_train = torch.cat([grid, x_train], dim=-1)
        x_test = torch.cat([grid_test, x_test], dim=-1)
        y_train = torch.cat([dataset.f_observed for dataset in datasets[:ntrain]]).permute(0, 2, 1)
        y_test = torch.cat([dataset.f_observed for dataset in datasets[ntrain:]]).permute(0, 2, 1)
        torch.save({'x_train': x_train, 'x_test': x_test,
                    'y_train': y_train, 'y_test': y_test}, Path(data_dir) / 'toy_transcriptomics.pt')
