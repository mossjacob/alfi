import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lafomo",
    version="0.0.3",
    author="Jacob Moss",
    author_email="cob.mossy@gmail.com",
    description="A Latent Force Model library with variational and MCMC support for non-linear functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JCobbles/reggae",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)