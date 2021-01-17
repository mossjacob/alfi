# REGGaE: Regulation of Gene Expression

[![Latest PyPI Version][pb]][pypi] [![PyPI Downloads][db]][pypi]

[pb]: https://img.shields.io/pypi/v/tf-reggae.svg
[pypi]: https://pypi.org/project/tf-reggae/

[db]: https://img.shields.io/pypi/dm/tf-reggae?label=pypi%20downloads

This library implements several Latent Force Models of transcriptional regulation. These are all implemented in building blocks simplifying the creation of novel models or combinations.

Both analytical and MCMC-based methods are implemented.

## Installation

`pip install tf-reggae`


## Examples

### Analytical - Linear, Single TF

Open `examples/analytical.ipynb` as a Jupyter notebook and run the cells. This is a replication of the analytical Linear Latent Force Model from Lawrence et al., 2006 (https://papers.nips.cc/paper/3119-modelling-transcriptional-regulation-using-gaussian-processes.pdf)

The dataset required is small and is available preprocessed here:
- https://drive.google.com/drive/folders/1Tg_3SlKbdv0pDog6k2ys0J79e1-vgRyd?usp=sharing

### MCMC - Non-linear, Multi TF

Open `examples/nonlinear_multitf.ipynb` as a Jupyter notebook.

- https://drive.google.com/drive/folders/12onIIO1_nt2pHb-HAMB7gEFjOYdrfR2c?usp=sharing