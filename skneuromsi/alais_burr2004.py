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


def auditory_estimator(auditory_sigma, possible_locations, auditory_location):
    """Computes the auditory estimate.

    Parameters
    ----------
    auditory_sigma: ``float``
        Standard deviation of the auditory estimate.
    possible_locations: ``ndarray``
        Numpy array containing all the possible locations where the stimulus
        could be delivered.
    auditory_location: ``float``
        Location in which the auditory stimulus is delivered.

    Returns
    ----------
    auditory_estimate: ``ndarray``
        Numpy array containing the estimated location in which the auditory
        stimulus was delivered.
    """

    sigma = auditory_sigma
    location = auditory_location
    plocations = possible_locations

    auditory_estimate = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )

    return auditory_estimate


def visual_estimator(possible_locations, visual_sigma, visual_location):
    """Computes the visual estimate.

    Parameters
    ----------
    visual_sigma: ``float``
        Standard deviation of the visual estimate.
    possible_locations: ``ndarray``
        Numpy array containing all the possible locations where the stimulus
        could be delivered.
    visual_location: ``float``
        Location in which the visual stimulus is delivered.

    Returns
    ----------
    visual_estimate: ``ndarray``
        Numpy array containing the estimated location in which the visual
        stimulus was delivered.
    """

    plocations = possible_locations
    location = visual_location
    sigma = visual_sigma

    visual_estimate = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )

    return visual_estimate


def multisensory_estimator(
    possible_locations,
    visual_location,
    auditory_location,
    visual_weight,
    auditory_weight,
    multisensory_sigma,
    auditory_estimator,
    visual_estimator,
):
    """
    Computes the multisensory estimate.

    Parameters
    ----------
    possible_locations: ``ndarray``
        Numpy array containing all the possible locations where the stimulus
        could be delivered.
    visual_location: ``float``
        Location in which the visual stimulus is delivered.
    auditory_location: ``float``
        Location in which the auditory stimulus is delivered.
    visual_weight: ``float``
        Relative weight of the visual modality.
    auditory_weight: ``float``
        Relative weight of the auditory modality.
    multisensory_sigma: ``float``
        Standard deviation of the multisensory estimate.
    auditory_estimator: ``ndarray``
        Results of the auditory estimator. Numpy array containing
        the estimated location in which the auditory stimulus
        was delivered.
    visual_estimator: ``ndarray``
        Results of the visual estimator. Numpy array containing
        the estimated location in which the visual stimulus
        was delivered.

    Returns
    ----------
    res: ``dict``
        Results of the multisensory integration model.
        Includes these fields:

        * *"auditory"*: Auditory estimate
        * *"visual"*: Visual estimate
        * *"multisensory"*: Multisensory estimate

    """

    sigma = multisensory_sigma

    location = (
        visual_weight * visual_location + auditory_weight * auditory_location
    )
    plocations = possible_locations

    multisensory_res = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )

    res = {
        "auditory": auditory_estimator,
        "visual": visual_estimator,
        "multisensory": multisensory_res,
    }

    return res


# ===============================================================================
# CLASSES
# ===============================================================================


@core.neural_msi_model
class AlaisBurr2004:
    """Class that implements the Near-optimal Bimodal Integration
    employed by Alais and Burr to reproduce the Ventriloquist Effect [1]_.

    Attributes
    ----------
    possible_locations: ``skneuromsi.hparameter``
        All the possible locations where the stimulus
        could be delivered.
    auditory_sigma: ``skneuromsi.hparameter``
        Standard deviation of the auditory estimate.
    visual_sigma: ``skneuromsi.hparameter``
        Standard deviation of the visual estimate.
    auditory_weight: ``skneuromsi.internal``
        Relative weight of the auditory modality.
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
    .. [1] D. Alais and D. Burr, “The Ventriloquist Effect Results from
        Near-Optimal Bimodal Integration,” Current Biology, vol. 14, no. 3,
        pp. 257-262, Feb. 2004, doi: 10.1016/j.cub.2004.01.029.
    """

    # hiper parameters
    possible_locations = core.hparameter(
        factory=lambda: np.arange(-20, 20, 0.01)
    )
    auditory_sigma = core.hparameter(default=3.0)
    visual_sigma = core.hparameter(default=3.0)

    # internals
    auditory_weight = core.internal()

    @auditory_weight.default
    def _auditory_weight_default(self):
        return self.visual_sigma ** 2 / (
            self.auditory_sigma ** 2 + self.visual_sigma ** 2
        )

    visual_weight = core.internal()

    @visual_weight.default
    def _visual_weights_default(self):
        return self.auditory_sigma ** 2 / (
            self.visual_sigma ** 2 + self.auditory_sigma ** 2
        )

    multisensory_sigma = core.internal()

    @multisensory_sigma.default
    def _multisensory_sigma_default(self):
        return np.sqrt(
            (self.visual_sigma ** 2 * self.auditory_sigma ** 2)
            / (self.auditory_sigma ** 2 + self.visual_sigma ** 2)
        )

    # estimators
    stimuli = [auditory_estimator, visual_estimator]
    integration = multisensory_estimator
