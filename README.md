# LaFoMo: Latent Force Models

> _Don't miss out!_

[![Latest PyPI Version][pb]][pypi] [![PyPI Downloads][db]][pypi] [![Documentation Status](https://readthedocs.org/projects/lafomo/badge/?version=latest)](https://lafomo.readthedocs.io/en/latest/?badge=latest)


[pb]: https://img.shields.io/pypi/v/lafomo.svg
[pypi]: https://pypi.org/project/lafomo/

[db]: https://img.shields.io/pypi/dm/lafomo?label=pypi%20downloads


### Implement Latent Force Models in under 10 lines of code!

This library implements several Latent Force Models. These are all implemented in building blocks simplifying the creation of novel models or combinations.

We support analytical (exact) inference in addition to inducing point approximations for non-linear LFMs written in PyTorch. Furthermore, an MCMC-based component library based on TensorFlow Probability is included for a fully Bayesian treatment of ODE parameters.

This library was previously known as REGGaE: Regulation of Gene Expression.

## Installation

`pip install lafomo`



## Documentation


See Jupyter notebook in the documentation [here](https://lafomo.readthedocs.io/en/latest/notebooks_list.html). The docs contain linear, non-linear (both variational and MCMC methods), and partial Latent Force Models. The notebooks contain complete examples such as a replication of the analytical Linear Latent Force Model from [Lawrence et al., 2006](https://papers.nips.cc/paper/3119-modelling-transcriptional-regulation-using-gaussian-processes.pdf)



