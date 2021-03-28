from gpytorch.mlls import ExactMarginalLogLikelihood

from lafomo.configuration import VariationalConfiguration
from lafomo.models import ExactLFM
from lafomo.plot import Plotter
from lafomo.trainers import ExactTrainer



def build_exact(dataset):
    model = ExactLFM(dataset, dataset.variance.reshape(-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)

    loss_fn = ExactMarginalLogLikelihood(model.likelihood, model)

    trainer = ExactTrainer(model, optimizer, dataset, loss_fn=loss_fn)
    plotter = Plotter(model, dataset.gene_names)
    model.likelihood.train()

    return model, trainer, plotter
