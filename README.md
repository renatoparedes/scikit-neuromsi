# scikit-neuromsi

[![scikit-neuromsi](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/skneuromsi_ci.yml/badge.svg)](https://github.com/renatoparedes/scikit-neuromsi/actions/workflows/skneuromsi_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/scikit-neuromsi/badge/?version=latest)](https://scikit-neuromsi.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/renatoparedes/scikit-neuromsi/badge.svg?branch=main)](https://coveralls.io/github/renatoparedes/scikit-neuromsi?branch=main)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Scikit-neuromsi** is an open-source Python framework that simplifies the implementation of neurocomputational models of multisensory integration.

## Motivation

Research on the the neural process by which unisensory signals are combined to form a significantly different multisensory response has grown exponentially in the recent years. Nevertheless, there is as yet no unified theoretical approach to multisensory integration. We believe that building a framework for multisensory integration modelling would greatly contribute to originate a unifying theory that narrows the gap between neural and behavioural multisensory responses. 

## Contact
Renato Paredes (paredesrenato92@gmail.com)

## Features

**Scikit-neuromsi** currently has three classes which implement neurocomputational 
models of multisensory integration.

The available modules are:

- **alais_burr2004**: implements the near-optimal bimodal integration
    employed by Alais and Burr (2004) to reproduce the Ventriloquist Effect.

- **ernst_banks2002**: implements the visual-haptic maximum-likelihood
    integrator employed by Ernst and Banks (2002) to reproduce the visual-haptic task.

- **kording2007**: implements the Bayesian Causal Inference model for
    Multisensory Perception employed by Kording et al. (2007) to reproduce
    the Ventriloquist Effect.

In addition, there is a **core** module with features to facilitate the implementation of new models of multisensory integration.

## Requirements

You need Python 3.9+ to run scikit-neuromsi.

## Installation

Run the following command:

        $ pip install scikit-neuromsi 

or clone this repo and then inside the local directory execute:

        $ pip install -e .