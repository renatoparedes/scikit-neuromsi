#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""
Implementation of multisensory integration neurocomputational models in Python.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..core import SKNMSIMethodABC

# =============================================================================
# FUNCTIONS
# =============================================================================


class AlaisBurr2004(SKNMSIMethodABC):
    """Zaraza.


    References
    ----------
    :cite:p:`cuppini2017biologically`

    """

    _model_name = "AlaisBurr2004"
    _model_type = "MLE"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_sigma", "template": "${mode0}_sigma"},
        {"target": "visual_sigma", "template": "${mode1}_sigma"},
    ]

    _run_output = [
        {"target": "auditory", "template": "${mode0}"},
        {"target": "visual", "template": "${mode1}"},
        {"target": "auditory_weight", "template": "${mode0}_weight"},
        {"target": "visual_weight", "template": "${mode1}_weight"},
    ]

    def __init__(
        self,
        *,
        mode0="auditory",
        mode1="visual",
        position_range=(-20, 20),
        position_res=0.01,
        time_range=(1, 1),
        time_res=1,
    ):

        self._mode0 = mode0
        self._mode1 = mode1
        self._position_range = position_range
        self._position_res = float(position_res)
        self._time_range = time_range
        self._time_res = float(time_res)

    # PROPERTY ================================================================

    @property
    def mode0(self):
        return self._mode0

    @property
    def mode1(self):
        return self._mode1

    @property
    def time_range(self):
        return self._time_range

    @property
    def time_res(self):
        return self._time_res

    @property
    def position_range(self):
        return self._position_range

    @property
    def position_res(self):
        return self._position_res

    # Model methods
    def unisensory_estimator(
        self, unisensory_sigma, unisensory_position, possible_locations
    ):

        sigma = unisensory_sigma
        location = unisensory_position
        plocations = possible_locations

        unisensory_estimate = (
            1 / np.sqrt(2 * np.pi * np.square(sigma))
        ) * np.exp(
            -1 * (((plocations - location) ** 2) / (2 * np.square(sigma)))
        )

        return unisensory_estimate

    def weight_calculator(self, target_sigma, reference_sigma):
        target_weight = np.square(reference_sigma) / (
            np.square(target_sigma) + np.square(reference_sigma)
        )
        return target_weight

    def multisensory_estimator(
        self,
        unisensory_position_a,
        unisensory_position_b,
        unisensory_weight_a,
        unisensory_weight_b,
        multisensory_sigma,
        possible_locations,
    ):

        sigma = multisensory_sigma

        location = (
            unisensory_weight_a * unisensory_position_a
            + unisensory_weight_b * unisensory_position_b
        )
        plocations = possible_locations

        multisensory_estimate = (
            1 / np.sqrt(2 * np.pi * np.square(sigma))
        ) * np.exp(
            -1 * (((plocations - location) ** 2) / (2 * np.square(sigma)))
        )

        return multisensory_estimate

    # Model run
    def run(
        self,
        *,
        auditory_position=-5,
        visual_position=5,
        auditory_sigma=3.0,
        visual_sigma=3.0,
    ):
        possible_locations = np.arange(
            self._position_range[0],
            self._position_range[1],
            self._position_res,
        )

        #        auditory_position = int(possible_locations.size / 2) if auditory_position is None else auditory_position
        #        visual_position = int(possible_locations.size / 2) if visual_position is None else visual_position

        auditory_estimate = self.unisensory_estimator(
            auditory_sigma, auditory_position, possible_locations
        )
        visual_estimate = self.unisensory_estimator(
            visual_sigma, visual_position, possible_locations
        )

        multisensory_sigma = np.sqrt(
            (np.square(visual_sigma) * np.square(auditory_sigma))
            / (np.square(auditory_sigma) + np.square(visual_sigma))
        )

        auditory_weight = self.weight_calculator(auditory_sigma, visual_sigma)

        visual_weight = self.weight_calculator(visual_sigma, auditory_sigma)

        multisensory_estimate = self.multisensory_estimator(
            auditory_position,
            visual_position,
            auditory_weight,
            visual_weight,
            multisensory_sigma,
            possible_locations,
        )

        response = {
            "auditory": auditory_estimate,
            "visual": visual_estimate,
            "multi": multisensory_estimate,
        }
        return response, {
            "auditory_weight": auditory_weight,
            "visual_weight": visual_weight,
        }
