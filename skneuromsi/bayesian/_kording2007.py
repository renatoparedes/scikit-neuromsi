#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""Implementation of multisensory integration models in Python."""

# =============================================================================
# IMPORTS
# =============================================================================
import copy

import numpy as np

from ..core import SKNMSIMethodABC

# =============================================================================
# FUNCTIONS
# =============================================================================


class Kording2007(SKNMSIMethodABC):
    r"""
    Bayesian Causal Inference model for multisensory integration.

    This model based on Kording et al. (2007) uses Bayesian principles to
    infer whether two unimodal signals come from a common cause or
    different causes. It combines auditory and visual signals and evaluates
    the probability of a common cause based on the observed signals.

    This implementation is inspired on the Matlab version of the BCI Toolbox
    (Zhu, Beierholm & Shams, 2024).


    References
    ----------
    :cite:p:`kording2007causal`
    :cite:p:`zhu2024bci`

    Notes
    -----
    The Bayesian Causal Inference model uses the following formulation:

    .. math::
       p(C \mid x_{1}, x_{2}) = \frac{p(x_{1}, x_{2} \mid C), p(C)}{p(x_{1},
       x_{2})}

    where :math:`x_{1}` and :math:`x_{2}` are two unimodal signals, and
    :math:`C` is a binary variable representing the number of causes in the
    environment.

    The posterior probability of the signals having a single cause in the
    environment is defined as follows:

    .. math::
        p(C = 1 \mid x_{1}, x_{2}) = \frac{p(x_{1}, x_{2} \mid C=1),
        p(C=1)}{p(x_{1}, x_{2} \mid C=1) \, p(C=1) + p(x_{1}, x_{2} \mid
        C=2), (1 - p(C=1))}

    The likelihood is computed as:

    .. math::
        p(x_{1}, x_{2} \mid C = 1) = \int\int p(x_{1}, x_{2} \mid X) p(X) dX

    Here, :math:`p(C = 1)` is the prior probability of a common cause
    (default is 0.5). :math:`X` denotes the attributes of the stimuli
    (e.g., distance), which are then represented in the nervous system as
    :math:`x_{1}` and :math:`x_{2}`.

    These equations show that the inference of a common cause of
    two unisensory signals is computed by combining the likelihood and prior
    of signals having a common cause. A higher likelihood occurs if the
    two unisensory signals are similar, which in turn increases the
    probability of inferring that the signals have a common cause.

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
    _output_mode = "multi"

    def __init__(
        self,
        *,
        n=1,
        mode0="auditory",
        mode1="visual",
        position_range=(-42, 42),
        position_res=1.7142857142857142,
        time_range=(0, 1),
        time_res=1,
        seed=None,
    ):
        """
        Initializes the Kording 2007 model.

        Parameters
        ----------
        n : int
            Number of simulations to run.
        mode0 : str
            The name for the first sensory modality (e.g., "auditory").
        mode1 : str
            The name for the second sensory modality (e.g., "visual").
        position_range : tuple of float
            The range of positions to consider for estimation. E.g., (-42, 43).
        position_res : float
            The resolution of positions to consider for estimation. E.g., 1.7.
        time_range : tuple of int
            The range of time steps to consider. E.g., (1, 1).
        time_res : int
            The resolution of time steps. E.g., 1.
        seed : int or None
            Seed for the random number generator.
            If None, the random number generator will not be seeded.
        """
        self._n = n
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
        """Returns the name of the first sensory modality.

        Returns
        -------
        str
            The name of the first sensory modality.
        """
        return self._mode0

    @property
    def mode1(self):
        """Returns the name of the second sensory modality.

        Returns
        -------
        str
            The name of the second sensory modality.
        """
        return self._mode1

    @property
    def n(self):
        """Returns the number of simulations.

        Returns
        -------
        int
            The number of simulations to run.
        """
        return self._n

    @property
    def time_range(self):
        """Returns the range of time steps considered.

        Returns
        -------
        tuple of int
            The range of time steps.
        """
        return self._time_range

    @property
    def time_res(self):
        """Returns the resolution of time steps considered.

        Returns
        -------
        int
            The resolution of time steps.
        """
        return self._time_res

    @property
    def position_range(self):
        """Returns the range of positions considered for estimation.

        Returns
        -------
        tuple of float
            The range of positions. E.g., (-42, 43).
        """
        return self._position_range

    @property
    def position_res(self):
        """Returns the resolution of positions considered for estimation.

        Returns
        -------
        float
            The resolution of positions. E.g., 1.7142857142857142.
        """
        return self._position_res

    @property
    def random(self):
        """Returns the random number generator.

        Returns
        -------
        numpy.random.Generator
            The random number generator.
        """
        return self._random

    # Model methods

    def input_computation(self, unisensory_position, unisensory_var, noise):
        """Computes the unisensory input.

        Unisensory input considers the position estimate and variance.
        In this implementation noise is optional.

        Parameters
        ----------
        unisensory_position : float
            The position estimate for the sensory modality.
        unisensory_var : float
            The variance of the sensory modality.
        noise : bool
            Whether to include noise in the computation.

        Returns
        -------
        float
            The computed input for the sensory modality.
        """
        if noise is False:
            return unisensory_position + np.sqrt(unisensory_var)
        return unisensory_position + np.sqrt(
            unisensory_var
        ) * self.random.standard_normal(self.n)

    def unisensory_estimator(
        self,
        unisensory_var,
        unisensory_var_hat,
        prior_var,
        prior_mu,
        unisensory_input,
    ):
        """Estimates the unisensory posterior given the input and prior.

        Parameters
        ----------
        unisensory_var : float
            The variance of the sensory modality.
        unisensory_var_hat : float
            The estimated variance of the sensory modality.
        prior_var : float
            The prior variance of the cause.
        prior_mu : float
            The prior mean of the cause.
        unisensory_input : float
            The input value for the sensory modality.

        Returns
        -------
        float
            The posterior estimate of the sensory modality.
        """
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
        """Estimates the multisensory posterior using Bayesian inference.

        Parameters
        ----------
        auditory_estimate : float
            The auditory estimate of the sensory modality.
        visual_estimate : float
            The visual estimate of the sensory modality.
        auditory_var : float
            The variance of the auditory modality.
        visual_var : float
            The variance of the visual modality.
        prior_var : float
            The prior variance of the cause.
        prior_mu : float
            The prior mean of the cause.
        multisensory_var : float
            The variance of the multisensory estimate.
        multisensory_var_hat : float
            The estimated variance of the multisensory estimate.
        n : int
            Number of simulations.
        auditory_ind_var : float
            The variance of the independent auditory estimate.
        visual_ind_var : float
            The variance of the independent visual estimate.
        p_common : float
            The prior probability of a common cause.
        strategy : str
            The strategy for model selection
            ("selection", "averaging", "matching").
        possible_locations : np.ndarray
            The possible positions to consider for estimation.
        auditory_input : float
            The input value for the auditory modality.
        visual_input : float
            The input value for the visual modality.

        Returns
        -------
        dict
            A dictionary with keys:
            - "auditory": The auditory posterior estimate.
            - "visual": The visual posterior estimate.
            - "multi": The multisensory posterior estimate.
            - "pc": The posterior probability of a common cause.
        """
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
                match = 1 - self.random.random(n)
                auditory_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * auditory_hat_ind
                visual_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * visual_hat_ind

        # Prediction of location estimates
        step = self._position_res
        edges = possible_locations - step / 2
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

    def set_random(self, rng):
        """Sets the random number generator.

        Parameters
        ----------
        rng : numpy.random.Generator
            The random number generator to set.
        """
        self._random = rng

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
        noise=True,
        causes_kind="count",
        dimension="space",
    ):
        """Runs the Bayesian causal inference model.

        Parameters
        ----------
        auditory_position : float
            The position of the auditory stimulus.
        visual_position : float
            The position of the visual stimulus.
        auditory_sigma : float
            The standard deviation of the auditory stimulus.
        visual_sigma : float
            The standard deviation of the visual stimulus.
        p_common : float
            The prior probability of a common cause.
        prior_sigma : float
            The standard deviation of the prior cause.
        prior_mu : float
            The mean of the prior cause.
        strategy : str
            The strategy for model selection
            ("selection", "averaging", "matching").
        noise : bool
            Whether to include noise in the computation.
        causes_kind : str
            The type of cause to calculate ("count" or "prob").
        dimension : str
            The dimension to run the model ("space" or "time").

        Returns
        -------
        tuple
            A tuple containing:
            - dict: Response dictionary with "auditory", "visual",
            and "multi" keys.
            - dict: Extra information with "mean_p_common_cause",
            "p_common_cause", and "causes_kind".
        """
        # Data holder
        pos_range = np.arange(
            self.position_range[0], self.position_range[1], self.position_res
        )
        t_range = np.arange(
            self.time_range[0], self.time_range[1], self.time_res
        )

        z_2d = np.zeros((t_range.size, pos_range.size))
        auditory_res, visual_res, multi_res = (
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
        )

        del z_2d

        # Dimension setup
        if dimension == "space":
            possible_locations = pos_range
        elif dimension == "time":
            possible_locations = t_range
        else:
            possible_locations = None

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
            auditory_position, auditory_var, noise=noise
        )
        visual_input = self.input_computation(
            visual_position, visual_var, noise=noise
        )

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

        if dimension == "space":
            auditory_res[0, :] = multisensory_estimate["auditory"]
            visual_res[0, :] = multisensory_estimate["visual"]
            multi_res[0, :] = multisensory_estimate["multi"]

        elif dimension == "time":
            auditory_res[:, 0] = multisensory_estimate["auditory"]
            visual_res[:, 0] = multisensory_estimate["visual"]
            multi_res[:, 0] = multisensory_estimate["multi"]

        response = {
            "auditory": auditory_res,
            "visual": visual_res,
            "multi": multi_res,
        }

        extra = {
            "mean_p_common_cause": np.average(multisensory_estimate["pc"]),
            "p_common_cause": multisensory_estimate["pc"],
            "causes_kind": causes_kind,
        }

        return response, extra

    def calculate_causes(self, mean_p_common_cause, causes_kind, **kwargs):
        """Calculates the causes of the stimuli.

        Parameters
        ----------
        mean_p_common_cause : float or np.ndarray
            The average probability of a common cause across simulations.
        causes_kind : str
            The type of cause to calculate ("count" or "prob").

        Returns
        -------
        int or float or None
            The number of causes if `causes_kind` is "count",
            the probability of a unique cause if `causes_kind` is "prob",
            or None if no valid cause type is specified.
        """
        # Determine the type of cause to calculate
        if causes_kind == "count":
            # evaluate the probability of perceiving a common cause
            if mean_p_common_cause > 0.5:
                causes = 1
            else:
                causes = 2

        elif causes_kind == "prob":
            # If calculating the probability of a unique cause,
            # assign the probability
            # of perceiving a common cause
            causes = mean_p_common_cause
        else:
            # If no valid cause type is specified, assign None
            causes = None
        return causes
