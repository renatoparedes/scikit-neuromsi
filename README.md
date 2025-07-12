# Scikit-NeuroMSI
![logo](https://raw.githubusercontent.com/renatoparedes/scikit-neuromsi/main/res/logo_banner.png)

<!-- BODY -->

[![scikit-neuromsi](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/ci.yml/badge.svg)](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/scikit-neuromsi/badge/?version=latest)](https://scikit-neuromsi.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/pypi/pyversions/scikit-neuromsi.svg)](https://pypi.org/project/scikit-neuromsi/)
[![PyPI](https://img.shields.io/pypi/v/scikit-neuromsi)](https://pypi.org/project/scikit-neuromsi/)
[![Coverage Status](https://coveralls.io/repos/github/renatoparedes/scikit-neuromsi/badge.svg?branch=main)](https://coveralls.io/github/renatoparedes/scikit-neuromsi?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![DOI](https://img.shields.io/badge/doi-10.1101/2025.05.26.656124-red)](https://doi.org/10.1101/2025.05.26.656124)
[![PyPI Downloads](https://static.pepy.tech/badge/scikit-neuromsi)](https://pepy.tech/projects/scikit-neuromsi)

**Scikit-NeuroMSI** is an open-source Python framework that simplifies the implementation of neurocomputational models of multisensory integration.

## Motivation

Research on the neural mechanisms underlying multisensory integration—where unisensory signals combine to produce distinct multisensory responses—has surged in recent years. Despite this progress, a unified theoretical framework for multisensory integration remains elusive. **Scikit-NeuroMSI** aims to bridge this gap by providing a standardized, flexible, and extensible platform for modeling multisensory integration, fostering the development of theories that connect neural and behavioral responses.

## Features

**Scikit-NeuroMSI** was designed to meet three fundamental requirements in the computational study of multisensory integration:

- **Modeling Standardization**: Standardized interface for implementing and analyzing different types of models. The package currently handles the following model families: Maximum Likelihood Estimation, Bayesian Causal Inference, and Neural Networks.

- **Data Processing Pipeline**: Multidimensional data processing across spatial dimensions (1D to 3D spatial coordinates), temporal sequences, and multiple sensory modalities (e.g., visual, auditory, touch).

- **Analysis Tools**: Integrated tools for parameter sweeping across model configurations, result visualization and export, and statistical analysis of model outputs.

In addition, there is a **core** module with features to facilitate the implementation of new models of multisensory integration.

## Requirements

You need Python 3.10+ to run scikit-neuromsi.

## Installation

Run the following command:

```bash
pip install scikit-neuromsi
```

or clone this repo and then inside the local directory execute:

```bash
pip install -e .
```

## Usage Examples

### Run models of multisensory integration

Simulate responses from existing models of multisensory integration (e.g.  audio-visual causal inference network from Cuppini et al. (2017)):

```python
from skneuromsi.neural import Cuppini2017

# Model setup
model_cuppini2017 = Cuppini2017(neurons=90, 
                                position_range=(0, 90))

# Model execution
res = model_cuppini2017.run(auditory_position=35, 
                            visual_position=52)

# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")
```
![model_result](https://raw.githubusercontent.com/renatoparedes/scikit-neuromsi/main/res/cuppini2017_output.png)

### Simulate experimental paradigms

Simulate multisensory integration experiments (e.g. causal inference under spatial disparity - "ventriloquist effect") using the parameter sweep tool:

```python
from skneuromsi.sweep import ParameterSweep
import numpy as np

# Experiment setup
spatial_disparities = np.array([-24, -12, -6, -3, 3, 6, 12, 24])

sp_cuppini2017 = ParameterSweep(model=model_cuppini2017,
                                target="visual_position",
                                repeat=1,
                                range=45 + spatial_disparities)

# Experiment run
res_sp_cuppini2017 = sp_cuppini2017.run(auditory_position=45,
                                        auditory_sigma=4.5,
                                        visual_sigma=3.5)

# Experiment results plot
ax1 = plt.subplot()
res_sp_cuppini2017.plot(kind="unity_report", label="Cuppini 2017", ax=ax1)
ax1.set_xlabel("visual position (deg)")
```
![unity_report_result](https://raw.githubusercontent.com/renatoparedes/scikit-neuromsi/main/res/causal_inference_output.png)

For more detailed examples and advanced usage, refer to the [Scikit-NeuroMSI Documentation](https://scikit-neuromsi.readthedocs.io/).

## Contribute to Scikit-NeuroMSI

We welcome contributions to Scikit-NeuroMSI! 

If you're a multisensory integration researcher, we encourage you to integrate your models directly into our package. If you're a software developer, we'd love your help in enhancing the overall functionality of Scikit-NeuroMSI. 

For detailed information on how to contribute ideas, report bugs, or improve the codebase, please refer to our [Contribuiting Guidelines](https://github.com/renatoparedes/scikit-neuromsi/blob/main/CONTRIBUTING.md).

## License

Scikit-NeuroMSI is under
[The 3-Clause BSD License](https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as
its copyright notices and the license’s disclaimers of warranty are maintained.

## How to cite?

If you want to cite Scikit-NeuroMSI, please use the following references:

> Paredes, R., Cabral, J. B., & Series, P. (2025). Scikit-NeuroMSI: A Generalized Framework for Modeling Multisensory Integration. bioRxiv, 2025-05. doi: https://doi.org/10.1101/2025.05.26.656124

>Paredes, R., Series, P., Cabral, J. (2023). Scikit-NeuroMSI: a Python framework for multisensory integration modelling. IX Congreso de Matematica Aplicada, Computacional e Industrial, 9, 545–548.