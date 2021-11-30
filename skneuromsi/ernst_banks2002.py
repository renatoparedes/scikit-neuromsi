#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-neuromsi Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
Implementation of multisensory integration neurocomputational models in Python.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from . import core

# =============================================================================
# FUNCTIONS
# =============================================================================


def haptic_estimator(haptic_sigma, possible_heights, haptic_height):
    """Computes the haptic estimate.

    Parameters
    ----------
    haptic_sigma: ``float``
        Standard deviation of the haptic estimate.
    possible_heights: ``ndarray``
        Numpy array containing all the possible heights of
        the stimulus.
    haptic_height: ``float``
        Height of the haptic stimulus.

    Returns
    ----------
    haptic_estimate: ``ndarray``
        Numpy array containing the estimated height of the haptic
        stimulus.
    """

    sigma = haptic_sigma
    height = haptic_height
    pheights = possible_heights

    haptic_estimate = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )

    return haptic_estimate


def visual_estimator(possible_heights, visual_sigma, visual_height):
    """Computes the visual estimate.

    Parameters
    ----------
    visual_sigma: ``float``
        Standard deviation of the visual estimate.
    possible_heights: ``ndarray``
        Numpy array containing all the possible heights of
        the stimulus.
    visual_height: ``float``
        Height of the visual stimulus.

    Returns
    ----------
    visual_estimate: ``ndarray``
        Numpy array containing the estimated height of the visual
        stimulus.
    """

    pheights = possible_heights
    height = visual_height
    sigma = visual_sigma

    visual_estimate = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )

    return visual_estimate


def multisensory_estimator(
    multisensory_sigma,
    possible_heights,
    visual_height,
    haptic_height,
    visual_weight,
    haptic_weight,
    haptic_estimator,
    visual_estimator,
):
    """
    Computes the multisensory estimate.

    Parameters
    ----------
    possible_heights: ``ndarray``
        Numpy array containing all the possible heights of the stimulus.
    visual_height: ``float``
        Height of the visual stimulus.
    haptic_height: ``float``
        Height of the haptic stimulus.
    visual_weight: ``float``
        Relative weight of the visual modality.
    haptic_weight: ``float``
        Relative weight of the haptic modality.
    multisensory_sigma: ``float``
        Standard deviation of the multisensory estimate.
    haptic_estimator: ``ndarray``
        Results of the haptic estimator. Numpy array containing
        the estimated height of the haptic stimulus.
    visual_estimator: ``ndarray``
        Results of the visual estimator. Numpy array containing
        the estimated height of the visual stimulus.

    Returns
    ----------
    res: ``dict``
        Results of the multisensory integration model.
        Includes these fields:

        * *"haptic"*: Haptic estimate
        * *"visual"*: Visual estimate
        * *"multisensory"*: Multisensory estimate

    """

    sigma = multisensory_sigma

    height = visual_weight * visual_height + haptic_weight * haptic_height
    pheights = possible_heights

    multisensory_res = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )

    return {
        "haptic": haptic_estimator,
        "visual": visual_estimator,
        "multisensory": multisensory_res,
    }


# ===============================================================================
# CLASSES
# ===============================================================================


@core.neural_msi_model
class ErnstBanks2002:
    """Class that implements the visual-haptic maximum-likelihood
    integrator employed by Ernst and Banks to reproduce the
    visual-haptic task [2]_.

    Attributes
    ----------
    possible_heights: ``skneuromsi.hparameter``
        All the possible heights of the stimulus.
    haptic_sigma: ``skneuromsi.hparameter``
        Standard deviation of the haptic estimate.
    visual_sigma: ``skneuromsi.hparameter``
        Standard deviation of the visual estimate.
    haptic_weight: ``skneuromsi.internal``
        Relative weight of the haptic modality.
    visual_weight: ``skneuromsi.internal``
        Relative weight of the visual modality.
    multisensory_sigma: ``skneuromsi.internal``
        Standard deviation of the multisensory estimate.
    stimuli: ``list`` of ``callable``
        List containing the functions employed for the
        computation of unisensory estimates.
    integration: ``callable``
        Function to compute the multisensory estimate.

    References
    ----------
    .. [2] M. O. Ernst and M. S. Banks, “Humans integrate
        visual and haptic information in a statistically
        optimal fashion,” Nature, vol. 415, no. 6870,
        pp. 429-433, Jan. 2002, doi: 10.1038/415429a.
    """

    # hiper parameters
    possible_heights = core.hparameter(factory=lambda: np.arange(49, 61, 0.01))
    haptic_sigma = core.hparameter(default=1)
    visual_sigma = core.hparameter(default=1)

    # internals
    haptic_weight = core.internal()

    @haptic_weight.default
    def _haptic_weight_default(self):
        return self.visual_sigma ** 2 / (
            self.haptic_sigma ** 2 + self.visual_sigma ** 2
        )

    visual_weight = core.internal()

    @visual_weight.default
    def _visual_weights_default(self):
        return self.haptic_sigma ** 2 / (
            self.visual_sigma ** 2 + self.haptic_sigma ** 2
        )

    multisensory_sigma = core.internal()

    @multisensory_sigma.default
    def _multisensory_sigma_default(self):
        return np.sqrt(
            (self.visual_sigma ** 2 * self.haptic_sigma ** 2)
            / (self.haptic_sigma ** 2 + self.visual_sigma ** 2)
        )

    # estimators
    stimuli = [haptic_estimator, visual_estimator]
    integration = multisensory_estimator
