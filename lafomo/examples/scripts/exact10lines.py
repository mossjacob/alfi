import torch

from lafomo.exact import AnalyticalLFM, Trainer
from lafomo.data_loaders import P53Data
from lafomo.plot.variational_plotters import Plotter

from matplotlib import pyplot as plt


if __name__ == '__main__':
    dataset = P53Data(data_dir='../../data', replicate=0)
    model = AnalyticalLFM(dataset, dataset.variance.reshape(-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)
    plotter = Plotter(model, dataset.gene_names)

    trainer.train(epochs=100)

    t_predict = torch.linspace(-1, 13, 80, dtype=torch.float64)
    m = plotter.plot_outputs(t_predict, t_scatter=dataset.t_observed, y_scatter=dataset.m_observed)
    plt.savefig('outputs.pdf')

    plotter.plot_latents(t_predict, num_samples=0)
    plt.savefig('latents.pdf')

    plotter.plot_kinetics()
    plt.savefig('kinetics.pdf')
