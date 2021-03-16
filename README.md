# LaFoMo: Latent Force Models

> _Don't miss out!_

[![Latest PyPI Version][pb]][pypi] [![PyPI Downloads][db]][pypi] [![Documentation Status](https://readthedocs.org/projects/lafomo/badge/?version=latest)](https://lafomo.readthedocs.io/en/latest/?badge=latest)


[pb]: https://img.shields.io/pypi/v/lafomo.svg
[pypi]: https://pypi.org/project/lafomo/

[db]: https://img.shields.io/pypi/dm/lafomo?label=pypi%20downloads


### Implement Latent Force Models in under 10 lines of code!

This library implements several Latent Force Models. These are all
implemented in building blocks simplifying the creation of novel models 
or combinations.

We support analytical (exact) inference in addition to inducing point approximations 
for non-linear LFMs written in PyTorch. Furthermore, an MCMC-based component library 
based on TensorFlow Probability is included for a fully Bayesian 
treatment of ODE parameters.

This library was previously known as REGGaE: Regulation of Gene Expression.

## Installation

`pip install lafomo`



## Examples: transcriptional regulation

### Analytical - Linear, Single TF

Open `examples/analytical.ipynb` as a Jupyter notebook and run the cells. This is a replication of the analytical Linear Latent Force Model from Lawrence et al., 2006 (https://papers.nips.cc/paper/3119-modelling-transcriptional-regulation-using-gaussian-processes.pdf)

The dataset required is small and is available preprocessed here:
- https://drive.google.com/drive/folders/1Tg_3SlKbdv0pDog6k2ys0J79e1-vgRyd?usp=sharing

### Variational - Non-linear Multi TF

Open `examples/variational.ipynb` as a Jupyter notebook and run the cells.

The dataset required is the same as the above.

### MCMC - Non-linear, Multi TF

Open `examples/mcmc_nonlinear.ipynb` as a Jupyter notebook.

- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100099
  
  - Download the file `GSE100099_RNASeqGEO.tsv.gz` 

