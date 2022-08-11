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


class Kording2007(SKNMSIMethodABC):
    """Zaraza.


    References
    ----------
    :cite:p:`cuppini2017biologically`

    """

    _model_name = "Kording2007"
    _model_type = "Bayesian"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_sigma", "template": "${mode0}_sigma"},
        {"target": "visual_sigma", "template": "${mode1}_sigma"},
    ]

    _run_output = [
        {"target": "auditory", "template": "${mode0}"},
        {"target": "visual", "template": "${mode1}"},
    ]

    def __init__(
        self,
        *,
        n=10000,
        mode0="auditory",
        mode1="visual",
        position=(-42, 42),
        position_res=1,
        time_res=1,
    ):

        self._n = n
        self._mode0 = mode0
        self._mode1 = mode1
        self._position = position
        self._position_res = float(position_res)
        self._time_res = float(time_res)

    # PROPERTY ================================================================

    @property
    def mode0(self):
        return self._mode0

    @property
    def mode1(self):
        return self._mode1

    @property
    def n(self):
        return self._n

    @property
    def position_res(self):
        return self._position_res

    @property
    def time_res(self):
        return self._time_res

    # Model methods

    def input_computation(self, unisensory_position, unisensory_var):
        return unisensory_position + np.sqrt(unisensory_var) * np.random.randn(
            self.n
        )

    def unisensory_estimator(
        self,
        unisensory_var,
        unisensory_var_hat,
        prior_var,
        prior_mu,
        unisensory_input,
    ):

        unisensory_hat_ind = (
            (unisensory_input / unisensory_var)
            + (np.ones(self.n) * prior_mu) / prior_var
        ) * unisensory_var_hat

        return unisensory_hat_ind

    def multisensory_estimator(
        self,
        auditory_estimate,
        visual_estimate,
        auditory_var,
        visual_var,
        prior_var,
        prior_mu,
        multisensory_var,
        multisensory_var_hat,
        n,
        auditory_ind_var,
        visual_ind_var,
        p_common,
        strategy,
        possible_locations,
        auditory_input,
        visual_input,
    ):

        # Inputs
        single_stim = np.sum(np.isnan([auditory_input, visual_input]))

        # Perceived location of causes
        auditory_hat_ind = auditory_estimate
        visual_hat_ind = visual_estimate
        multisensory_hat = (
            (auditory_input / auditory_var)
            + (visual_input / visual_var)
            + (np.ones(n) * prior_mu) / prior_var
        ) * multisensory_var_hat

        # Perceived distances
        dis_common = (
            (auditory_input - visual_input) ** 2 * prior_var
            + (auditory_input - prior_mu) ** 2 * visual_var
            + (visual_input - prior_mu) ** 2 * auditory_var
        )
        dis_auditory = (auditory_input - prior_mu) ** 2
        dis_visual = (visual_input - prior_mu) ** 2

        # Likelihood calculations
        likelihood_common = np.exp(-dis_common / (2 * multisensory_var)) / (
            2 * np.pi * np.sqrt(multisensory_var)
        )
        likelihood_auditory = np.exp(
            -dis_auditory / (2 * auditory_ind_var)
        ) / np.sqrt(2 * np.pi * auditory_ind_var)
        likelihood_visual = np.exp(
            -dis_visual / (2 * visual_ind_var)
        ) / np.sqrt(2 * np.pi * visual_ind_var)
        likelihood_indep = likelihood_auditory * likelihood_visual
        post_common = likelihood_common * p_common
        post_indep = likelihood_indep * (1 - p_common)
        pc = post_common / (post_common + post_indep)

        # Independent Causes
        if single_stim:
            auditory_hat = auditory_hat_ind
            visual_hat = visual_hat_ind
        else:
            # Decision Strategies
            # Model Selection
            if strategy == "selection":
                auditory_hat = (pc > 0.5) * multisensory_hat + (
                    pc <= 0.5
                ) * auditory_hat_ind
                visual_hat = (pc > 0.5) * multisensory_hat + (
                    pc <= 0.5
                ) * visual_hat_ind
            # Model Averaging
            elif strategy == "averaging":
                auditory_hat = (pc) * multisensory_hat + (
                    1 - pc
                ) * auditory_hat_ind
                visual_hat = (pc) * multisensory_hat + (
                    1 - pc
                ) * visual_hat_ind
            # Model Matching
            elif strategy == "matching":
                match = 1 - np.random.rand(n)
                auditory_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * auditory_hat_ind
                visual_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * visual_hat_ind

        # Prediction of location estimates
        step = possible_locations[1]
        edges = possible_locations[0] - step / 2
        edges = np.append(edges, edges[-1] + step)

        auditory_estimates = np.histogram(auditory_hat, edges)[0]
        visual_estimates = np.histogram(visual_hat, edges)[0]
        multisensory_estimates = np.histogram(multisensory_hat, edges)[0]

        pred_auditory = auditory_estimates / np.sum(auditory_estimates, axis=0)
        pred_visual = visual_estimates / np.sum(visual_estimates, axis=0)
        pred_multi = multisensory_estimates / np.sum(
            multisensory_estimates, axis=0
        )

        res = {
            "auditory": pred_auditory,
            "visual": pred_visual,
            "multi": pred_multi,
            "pc": pc,
        }

        return res

    def run(
        self,
        *,
        auditory_position=-15,
        visual_position=15,
        auditory_sigma=2.0,
        visual_sigma=10.0,
        p_common=0.5,
        prior_sigma=20.0,
        prior_mu=0,
        strategy="averaging",
    ):

        possible_locations = np.linspace(
            self._position[0], self._position[1], 50, retstep=True
        )

        visual_var = np.square(visual_sigma)
        auditory_var = np.square(auditory_sigma)
        prior_var = np.square(prior_sigma)

        multisensory_var = (
            auditory_var * visual_var
            + auditory_var * prior_var
            + visual_var * prior_var
        )

        auditory_ind_var = auditory_var + prior_var
        visual_ind_var = visual_var + prior_var

        multisensory_var_hat = 1 / (
            1 / auditory_var + 1 / visual_var + 1 / prior_var
        )

        auditory_var_hat = 1 / (1 / auditory_var + 1 / prior_var)

        visual_var_hat = 1 / (1 / visual_var + 1 / prior_var)

        auditory_input = self.input_computation(
            auditory_position, auditory_var
        )
        visual_input = self.input_computation(visual_position, visual_var)

        auditory_estimate = self.unisensory_estimator(
            auditory_var,
            auditory_var_hat,
            prior_var,
            prior_mu,
            auditory_input,
        )

        visual_estimate = self.unisensory_estimator(
            visual_var,
            visual_var_hat,
            prior_var,
            prior_mu,
            visual_input,
        )

        multisensory_estimate = self.multisensory_estimator(
            auditory_estimate,
            visual_estimate,
            auditory_var,
            visual_var,
            prior_var,
            prior_mu,
            multisensory_var,
            multisensory_var_hat,
            self.n,
            auditory_ind_var,
            visual_ind_var,
            p_common,
            strategy,
            possible_locations,
            auditory_input,
            visual_input,
        )

        response = {
            "auditory": multisensory_estimate["auditory"],
            "visual": multisensory_estimate["visual"],
            "multi": multisensory_estimate["multi"],
        }

        return response, {
            "mean_p_common_cause": np.average(multisensory_estimate["pc"]),
            "p_common_cause": multisensory_estimate["pc"],
        }


# TODO implement kde plot for p_common_cause for this model only:
# sns.kdeplot(res.e_.p_common_cause)
