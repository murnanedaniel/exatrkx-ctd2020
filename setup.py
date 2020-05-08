from setuptools import find_packages
from setuptools import setup
import fileinput
from setuptools.command.install import install 
import os


description="Use Graph Network to reconstruct tracks"

dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "matplotlib",
        "torch-scatter",
        'sklearn',
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',

        ]

if USE_CUDA: dependencies += "cupy"

setup(
    name="ExatrkX-CTD2020",
    version="0.0.1",
    description="Library for working with TrackML data to produce seeds and track labels.",
    long_description=description,
    author="Daniel Murnane, on behalf of Exa.Trkx Collaboration",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track finding", "tracking", "seeding", "GNN", "machine learning"],
    url="https://github.com/exatrkx/exatrkx-work",
    packages=find_packages(),
    install_requires=dependencies,
    cmdclass={'install': CustomInstall},
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
    ],
)