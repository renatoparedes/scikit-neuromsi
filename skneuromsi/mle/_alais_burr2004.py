#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..core import SKNMSIMethodABC

# =============================================================================
# FUNCTIONS
# =============================================================================


class AlaisBurr2004(SKNMSIMethodABC):
    r"""Optimal bimodal estimator model of Alais and Burr (2004).

    This model estimates multisensory integration by combining
    unisensory estimates with weights derived from the standard
    deviations of sensory modalities. It follows the Near-optimal Bimodal
    Integrator model described in Alais and Burr (2004) for
    integrating auditory and visual information.


    References
    ----------
    :cite:p:`alais2004ventriloquist`


    Notes
    -----
    The Near-optimal Bimodal Integrator for visual (V) and auditory (A) signals
    can be computed as:

    .. math::
        \hat{S}_{VA} = w_{V} \hat{S}_{V} + w_{A} \hat{S}_{A}

    where :math:`\hat{S}_{V}` and :math:`\hat{S}_{A}` are unimodal auditory
    and visual estimates, respectively, and :math:`\hat{S}_{VA}`
    is the multimodal estimate.

    In addition, :math:`w_{A}` and :math:`w_{V}` are the relative weights
    for each modality, defined as:

    .. math::
        w_{A} = \frac{\sigma_{V}^{2}}{\sigma_{A}^{2} + \sigma_{V}^{2}} \\
        w_{V} = \frac{\sigma_{A}^{2}}{\sigma_{V}^{2} + \sigma_{A}^{2}}

    where :math:`\sigma_{A}` and :math:`\sigma_{V}` are the
    standard deviations (or square roots of the variances) of each
    unimodal stimuli, respectively.

    These equations show that the optimal multisensory estimate combines
    the unisensory estimates weighted by their normalized reciprocal variances.

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
    _output_mode = "multi"

    def __init__(
        self,
        *,
        mode0="auditory",
        mode1="visual",
        position_range=(-20, 20),
        position_res=0.01,
        time_range=(1, 1),
        time_res=1,
        seed=None,
    ):
        """
        Initializes the Alais-Burr 2004 model.

        Parameters
        ----------
        mode0 : str
            The name for the first sensory modality (e.g., "auditory").
        mode1 : str
            The name for the second sensory modality (e.g., "visual").
        position_range : tuple of float
            The range of positions to consider for estimation. E.g., (-20, 20).
        position_res : float
            The resolution of positions to consider for estimation. E.g., 0.01.
        seed : int or None
            Seed for the random number generator.
            If None, the random number generator will not be seeded.
        """
        self._mode0 = mode0
        self._mode1 = mode1
        self._position_range = position_range
        self._position_res = float(position_res)
        self._time_range = time_range
        self._time_res = float(time_res)

        self.set_random(np.random.default_rng(seed=seed))

    # PROPERTY ================================================================

    @property
    def mode0(self):
        """
        Returns the name of the first sensory modality.

        Returns
        -------
        str
            The name of the first sensory modality.
        """
        return self._mode0

    @property
    def mode1(self):
        """
        Returns the name of the second sensory modality.

        Returns
        -------
        str
            The name of the second sensory modality.
        """
        return self._mode1

    @property
    def time_range(self):
        """
        Returns the range of time considered for estimation.

        Not used in this implementation.

        Returns
        -------
        tuple of float
            The range of time. E.g., (0, 100).
        """
        return self._time_range

    @property
    def time_res(self):
        """
        Returns the resolution of time considered for estimation.

        Not used in this implementation.

        Returns
        -------
        float
            The resolution of time. E.g., 0.01.
        """
        return self._time_res

    @property
    def position_range(self):
        """
        Returns the range of positions considered for estimation.

        Returns
        -------
        tuple of float
            The range of positions. E.g., (-20, 20).
        """
        return self._position_range

    @property
    def position_res(self):
        """
        Returns the resolution of positions considered for estimation.

        Returns
        -------
        float
            The resolution of positions. E.g., 0.01.
        """
        return self._position_res

    # Model methods
    def unisensory_estimator(
        self, unisensory_sigma, unisensory_position, possible_locations
    ):
        """
        Estimates the unisensory probability density function.

        Parameters
        ----------
        unisensory_sigma : float
            The standard deviation of the sensory modality.
        unisensory_position : float
            The position estimate of the sensory modality.
        possible_locations : numpy.ndarray
            The array of possible positions to evaluate.

        Returns
        -------
        numpy.ndarray
            The estimated probability density function
            for the given sensory modality.
        """
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
        """
        Calculates the weight of a sensory modality.

        Parameters
        ----------
        target_sigma : float
            The standard deviation of the target sensory modality.
        reference_sigma : float
            The standard deviation of the reference sensory modality.

        Returns
        -------
        float
            The weight of the target sensory modality.
        """
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
        """
        Estimates the multisensory probability density function.

        Parameters
        ----------
        unisensory_position_a : float
            The position estimate of the first sensory modality.
        unisensory_position_b : float
            The position estimate of the second sensory modality.
        unisensory_weight_a : float
            The weight of the first sensory modality.
        unisensory_weight_b : float
            The weight of the second sensory modality.
        multisensory_sigma : float
            The standard deviation of the multisensory estimate.
        possible_locations : numpy.ndarray
            The array of possible positions to evaluate.

        Returns
        -------
        numpy.ndarray
            The estimated multisensory probability density function.
        """
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
    def set_random(self, rng):
        """
        Sets the random number generator.

        Parameters
        ----------
        rng : numpy.random.Generator
            The random number generator to use.
        """
        self._random = rng

    def run(
        self,
        *,
        auditory_position=-5,
        visual_position=5,
        auditory_sigma=3.0,
        visual_sigma=3.0,
        noise=None,
    ):
        """
        Run the simulation of the Alais-Burr 2004 model.

        Parameters
        ----------
        auditory_position : float
            The position estimate for the auditory modality.
        visual_position : float
            The position estimate for the visual modality.
        auditory_sigma : float
            The standard deviation for the auditory modality.
        visual_sigma : float
            The standard deviation for the visual modality.
        noise : any, optional
            Additional noise to apply, if needed
            (not used in this implementation).

        Returns
        -------
        tuple
            A tuple containing:

            - response : dict
                A dictionary with keys 'auditory', 'visual',
                and 'multi' containing the auditory,
                visual, and multisensory estimates respectively.

            - extra : dict
                A dictionary with keys 'auditory_weight' and 'visual_weight'
                containing the weights for the auditory and visual
                modalities respectively.
        """
        possible_locations = np.arange(
            self._position_range[0],
            self._position_range[1],
            self._position_res,
        )

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
        extra = {
            "auditory_weight": auditory_weight,
            "visual_weight": visual_weight,
        }
        return response, extra
