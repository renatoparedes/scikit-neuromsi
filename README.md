# Scikit-NeuroMSI
![logo](https://raw.githubusercontent.com/renatoparedes/scikit-neuromsi/main/res/logo_banner.png)

<!-- BODY -->

[![scikit-neuromsi](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/ci.yml/badge.svg)](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/scikit-neuromsi/badge/?version=latest)](https://scikit-neuromsi.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/scikit-neuromsi)](https://pypi.org/project/scikit-neuromsi/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/renatoparedes/scikit-neuromsi/badge.svg?branch=main)](https://coveralls.io/github/renatoparedes/scikit-neuromsi?branch=main)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Scikit-NeuroMSI** is an open-source Python framework that simplifies the implementation of neurocomputational models of multisensory integration.

## Motivation

Research on the the neural process by which unisensory signals are combined to form a significantly different multisensory response has grown exponentially in the recent years. Nevertheless, there is as yet no unified theoretical approach to multisensory integration. We believe that building a framework for multisensory integration modelling would greatly contribute to originate a unifying theory that narrows the gap between neural and behavioural multisensory responses.

## Contact
Renato Paredes (paredesrenato92@gmail.com)

## Repository and Issues

https://github.com/renatoparedes/scikit-neuromsi


## License

Scikit-NeuroMSI is under
[The 3-Clause BSD License](https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as
its copyright notices and the license’s disclaimers of warranty are maintained.


## Features

**Scikit-NeuroMSI** was designed to meet three fundamental requirements in the computational study of multisensory integration:

- **Modeling Standardization**: Standardized interface for implementing and analyzing different types of models. The package currently handles the following model families: Maximum Likelihood Estimation, Bayesian Causal Inference, and Neural Networks.

- **Data Processing Pipeline**: Multidimensional data processing across spatial dimensions (1D to 3D spatial coordinates), temporal sequences, and multiple sensory modalities (e.g., visual, auditory, touch).

- **Analysis Tools**: Integrated tools for parameter sweeping across model configurations, result visualisation and export, and statistical analysis of model outputs.

In addition, there is a **core** module with features to facilitate the implementation of new models of multisensory integration.

## Requirements

You need Python 3.10+ to run scikit-neuromsi.

## Installation

Run the following command:

        $ pip install scikit-neuromsi

or clone this repo and then inside the local directory execute:

        $ pip install -e .