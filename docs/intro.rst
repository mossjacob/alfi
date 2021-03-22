------------
Introduction
------------

LaFoMo is a package for building Latent Force Models in Python. LaFoMo supports exact and variational inference for linear and non-linear LFMs respectively using `PyTorch <https://pytorch.org/>`_. In addition, we also support MCMC models for a full Bayesian treatment, and this uses `TensorFlow <https://www.tensorflow.org/>`_. LaFoMo was originally created and is now managed by `Jacob Moss <https://www.cl.cam.ac.uk/~jm2311/>`_.
The full list of contributors (in alphabetical order) is Bianca Dumitascu, Felix Opolka, Jeremy England, and Pietro Lio. If you are interested in contributing your Latent Force Models to this library please feel free to submit pull requests or contact us.

Install
-------

LaFoMo can be installed running ``pip install lafomo``. This also installs required dependencies including PyTorch and TensorFlow.

.. Version history is documented `here <https://github.com/mossjacob/lafomo/blob/master/RELEASE.md>`_.


Getting Started
---------------
Get started with our `examples and tutorials <notebooks_list.html>`_.


Which models are implemented?
-----------------------------
LaFoMo implements both linear and non-linear Latent Force Models in both the temporal setting (ODE-based) and spatio-temporal (PDE-based). Given that there are a number of methods for dealing with approximate inference, we provide a variational inference module in addition to a number of MCMC samplers. The options are currently:

Linear ODE
""""""""""

Dependencies:

* `GPyTorch <https://gpytorch.ai/>`_

Linear ODEs can have exact solutions for the GP covariance. LaFoMo has a full implementation of the model by `Lawrence et al. <http://papers.nips.cc/paper/3119-modelling-transcriptional-regulation-using-gaussian-processes.pdf>`_ [1] using the :class:`lafomo.models.ExactLFM` class. See the `notebook here <notebooks/linear/exact.html>`_ for the example implementation.


Non-linear ODE
""""""""""""""

Dependencies:

* `GPyTorch <https://gpytorch.ai/>`_

For non-linear ODEs which result in non-Gaussian likelihoods, we have two devoted modules. For variational inference, the :class:`lafomo.VariationalLFM` superclass is the main building block for such models. The implementation of the GP uses the method of `Hensman et al. <http://proceedings.mlr.press/v38/hensman15.pdf>`_ [3]. See the `notebook here <notebooks/linear/variational.html>`_ for a linear implementation, and `here <notebooks/nonlinear/variational.html>`_ for an implementation of a non-linear version of the *Lawrence et al.* [1] model.

The :class:`lafomo.mcmc` module implements several MCMC samplers suited for LFMs, such as the :class:`lafomo.mcmc.samplers.LatentGPSampler` which jointly samples the posterior GP in addition to the covariance kernel parameters. In addition, LaFoMo has a full implementation of such a non-linear LFM based on `Titsias et al. <https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-53>`_ [2]: :class:`lafomo.mcmc.modules.TranscriptionRegulationLFM`  .


Non-linear PDE
""""""""""""""

Dependencies:

* `GPyTorch <https://gpytorch.ai/>`_

* `FEniCS <https://fenicsproject.org/download/>`_ (this only tested on MacOS/Linux),

* `torch-fenics <https://github.com/barkm/torch-fenics/>`_.



Citing GPflow
-------------

To cite LaFoMo, please reference the `arxiv paper <https://arxiv.org/abs/2010.02555>`_. Sample BibTeX is given below:

.. code-block:: bib

    @article{moss2020gene,
      title={Gene Regulatory Network Inference with Latent Force Models},
      author={Moss, Jacob and Li{\'o}, Pietro},
      journal={arXiv preprint arXiv:2010.02555},
      year={2020}
    }


References
----------
[1] Modelling transcriptional regulation using Gaussian processes
Lawrence, N.D., Sanguinetti, G., and Rattray, M.
Advances in Neural Information Processing Systems, 1639-1647, 2006.

[2] Identifying targets of multiple co-regulating transcription factors from expression time-series by Bayesian model comparison.
Titsias, M.K., Honkela, A., Lawrence, N.D. and Rattray, M.
BMC systems biology, 6(1), pp.1-21, 2012.

[3] Scalable Variational Gaussian Process Classification
J Hensman, A G de G Matthews, Z Ghahramani
Proceedings of AISTATS 18, 2015.


Acknowledgements
----------------

Jacob Moss is supported by a GSK grant.