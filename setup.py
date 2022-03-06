import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alfi",
    version="1.0.0",
    author="Jacob Moss",
    author_email="cob.mossy@gmail.com",
    description="An approximate latent force model library with variational inference for non-linear ODEs and PDEs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mossjacob/lafomo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.20',
        'tqdm>=4.62',
        'scvelo>=0.2.4',
        'torch>=1.7.1',
        'torchdiffeq>=0.2.0',
        'pandas>=1.2.1',
        'matplotlib',
        'gpytorch>=1.3.1',
        'torchcubicspline @ https://github.com/patrick-kidger/torchcubicspline/tarball/master',
    ]
)
