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
import numpy as np

from ..core import SKNMSIMethodABC

# =============================================================================
# FUNCTIONS
# =============================================================================


class Zhu2024(SKNMSIMethodABC):
    """
    Multidimensional Bayesian Causal Inference (BCI) model.

    This model based on Zhu et al. (2024) uses Bayesian principles to infer
    whether different sensory inputs (auditory and visual) share a common cause
    or arise from independent sources. It extends traditional BCI approaches by
    simultaneously processing multiple dimensions (numerosity and timing).

    This model is specifically designed to explain the
    Sound-Induced Flash Illusion, where a single visual flash accompanied by
    multiple auditory beeps is often perceived as more than one flash due to
    cross-modal integration.

    The implementation follows the mathematical formulation described in
    Zhu et al. (2024, 2025), and is tested to reproduce the modelling results
    described in Zhu et al. (2024).


    References
    ----------
    :cite:p:`zhu2024overlooked`
    :cite:p:`zhu2025crossmodal`

    """

    _model_name = "Zhu2024"
    _model_type = "Bayesian"
    _run_input = [
        {"target": "auditory_time", "template": "${mode0}_time"},
        {"target": "visual_time", "template": "${mode1}_time"},
        {"target": "auditory_time_sigma", "template": "${mode0}_time_sigma"},
        {"target": "visual_time_sigma", "template": "${mode1}_time_sigma"},
        {"target": "auditory_numerosity", "template": "${mode0}_numerosity"},
        {"target": "visual_numerosity", "template": "${mode1}_numerosity"},
        {
            "target": "auditory_numerosity_sigma",
            "template": "${mode0}_numerosity_sigma",
        },
        {
            "target": "visual_numerosity_sigma",
            "template": "${mode1}_numerosity_sigma",
        },
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
        numerosity_range=(0, 4),
        numerosity_res=0.1,
        time_range=(0, 1000),
        time_res=10,
        position_range=(0, 1),
        position_res=1,
        seed=None,
    ):
        """
        Initializes the Zhou 2025 model.

        Parameters
        ----------
        n : int
            Number of simulations to run (sample size).
        mode0 : str
            First sensory modality (default: "auditory").
        mode1 : str
            Second sensory modality (default: "visual").
        numerosity_range : tuple
            Range of possible numerosity values (min, max).
        numerosity_res : float
            Resolution (step size) for numerosity discretization.
        time_range : tuple
            Range of possible time values in milliseconds (min, max).
        time_res : float
            Resolution (step size) for time discretization in milliseconds.
        seed : int, optional
            Random seed for reproducibility.
        """
        self._n = n
        self._mode0 = mode0
        self._mode1 = mode1
        self._numerosity_range = numerosity_range
        self._numerosity_res = float(numerosity_res)
        self._time_range = time_range
        self._time_res = float(time_res)
        self._position_range = position_range
        self._position_res = float(position_res)
        self.set_random(np.random.default_rng(seed=seed))

    # PROPERTY ================================================================

    @property
    def mode0(self):
        """First sensory modality (typically auditory)."""
        return self._mode0

    @property
    def mode1(self):
        """Second sensory modality (typically visual)."""
        return self._mode1

    @property
    def n(self):
        """Number of simulations to run."""
        return self._n

    @property
    def numerosity_range(self):
        """Range of possible numerosity values (min, max)."""
        return self._numerosity_range

    @property
    def numerosity_res(self):
        """Resolution (step size) for numerosity discretization."""
        return self._numerosity_res

    @property
    def time_range(self):
        """Range of possible time values in milliseconds (min, max)."""
        return self._time_range

    @property
    def time_res(self):
        """Resolution (step size) for time discretization in milliseconds."""
        return self._time_res

    @property
    def position_range(self):
        """Returns the range of positions considered for estimation.

        Returns
        -------
        tuple of float
            The range of positions. E.g., (0, 1).
        """
        return self._position_range

    @property
    def position_res(self):
        """Returns the resolution of positions considered for estimation.

        Returns
        -------
        float
            The resolution of positions. E.g., 1.0.
        """
        return self._position_res

    @property
    def random(self):
        """Random number generator instance."""
        return self._random

    def set_random(self, rng):
        """
        Set the random number generator for stochastic simulations.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator instance.
        """
        self._random = rng

    # Model methods

    def input_computation(self, value, variance, noise):
        """
        Generate noisy sensory measurements by adding Gaussian noise.

        This simulates how sensory inputs are corrupted by
        independent Gaussian noise during neural processing.

        Parameters
        ----------
        value : float
            True stimulus value (e.g., actual number of flashes or beeps).
        variance : float
            Variance of the sensory noise.
        noise : bool
            If True, adds random noise; if False, adds deterministic noise.

        Returns
        -------
        float or ndarray
            Noisy sensory measurement(s).
        """
        if not noise:
            # For deterministic mode, add fixed offset (for debugging/testing)
            return value + np.sqrt(variance)
        # For stochastic mode, add Gaussian noise scaled by the variance
        return value + np.sqrt(variance) * self.random.standard_normal(self.n)

    def unisensory_estimator(self, value, sigma, prior_mu, prior_sigma):
        """
        Compute optimal Bayesian estimate for a single sensory modality.

        This implements the reliability-weighted combination of sensory
        evidence and prior expectations, following the Bayes optimal
        observer model.

        Parameters
        ----------
        value : float or ndarray
            Noisy sensory measurement(s).
        sigma : float
            Standard deviation of the sensory noise.
        prior_mu : float
            Mean of the prior distribution.
        prior_sigma : float
            Standard deviation of the prior distribution.

        Returns
        -------
        float or ndarray
            Optimal unisensory estimate(s).
        """
        # Convert standard deviations to variances
        var = sigma**2
        prior_var = prior_sigma**2

        # Optimal Bayesian combination: weighted average based on precision
        estimate = (value / var + prior_mu / prior_var) / (
            1 / var + 1 / prior_var
        )
        return estimate

    def multisensory_estimator(
        self,
        aud_num,
        vis_num,
        aud_time,
        vis_time,
        aud_num_sigma,
        vis_num_sigma,
        aud_time_sigma,
        vis_time_sigma,
        prior_num_mu,
        prior_num_sigma,
        prior_time_mu,
        prior_time_sigma,
        p_common,
        strategy,
        numerosity_possible,
        time_possible,
        est_aud_num_ind,
        est_vis_num_ind,
        est_aud_time_ind,
        est_vis_time_ind,
    ):
        """
        Computes the multisensory estimates for both numerosity and time.

        This method first computes the common and independent likelihoods for
        each dimension, combines them to compute the posterior probability of a
        common cause (pc), and then uses a decision strategy (selection,
        averaging, or matching) to combine the estimates.

        Parameters
        ----------
        aud_num, vis_num : float or ndarray
            Auditory and visual numerosity inputs.
        aud_time, vis_time : float or ndarray
            Auditory and visual time inputs.
        aud_num_sigma, vis_num_sigma : float
            Standard deviations for auditory and visual numerosity.
        aud_time_sigma, vis_time_sigma : float
            Standard deviations for auditory and visual time.
        prior_num_mu, prior_time_mu : float
            Prior means for numerosity and time.
        prior_num_sigma, prior_time_sigma : float
            Prior standard deviations for numerosity and time.
        p_common : float
            Prior probability of a common cause.
        strategy : str
            Decision strategy ("selection", "averaging", or "matching").
        numerosity_possible : ndarray
            Possible numerosity values (bins) for estimation.
        time_possible : ndarray
            Possible time values (bins) for estimation.
        est_aud_num_ind, est_vis_num_ind : ndarray
            Independent estimates for auditory and visual numerosity.
        est_aud_time_ind, est_vis_time_ind : ndarray
            Independent estimates for auditory and visual time.

        Returns
        -------
        dict
            Dictionary with keys:
              - "auditory_numerosity": Distribution for auditory numerosity.
              - "visual_numerosity": Distribution for visual numerosity.
              - "auditory_time": Distribution for auditory time.
              - "visual_time": Distribution for visual time.
              - "pc": Posterior probability of a common cause.
        """
        # Convert standard deviations to variances for easier calculations
        var_aud_num = aud_num_sigma**2
        var_vis_num = vis_num_sigma**2
        prior_var_num = prior_num_sigma**2

        var_aud_time = aud_time_sigma**2
        var_vis_time = vis_time_sigma**2
        prior_var_time = prior_time_sigma**2

        # Multisensory variances used in likelihood calculations for C=1
        multisensory_var_num = (
            var_aud_num * var_vis_num
            + var_aud_num * prior_var_num
            + var_vis_num * prior_var_num
        )
        multisensory_var_time = (
            var_aud_time * var_vis_time
            + var_aud_time * prior_var_time
            + var_vis_time * prior_var_time
        )

        # Independent variances for unisensory estimates
        aud_ind_var_num = var_aud_num + prior_var_num
        vis_ind_var_num = var_vis_num + prior_var_num
        aud_ind_var_time = var_aud_time + prior_var_time
        vis_ind_var_time = var_vis_time + prior_var_time

        # Calculate the squared Mahalanobis distance for the C=1 hypothesis
        dis_common_num = (
            (aud_num - vis_num) ** 2 * prior_var_num
            + (aud_num - prior_num_mu) ** 2 * var_vis_num
            + (vis_num - prior_num_mu) ** 2 * var_aud_num
        )

        # Compute the likelihood for C=1 in numerosity dimension
        likelihood_common_num = np.exp(
            -dis_common_num / (2 * multisensory_var_num)
        ) / (2 * np.pi * np.sqrt(multisensory_var_num))

        # Compute independent likelihoods for each modality
        likelihood_aud_num = np.exp(
            -((aud_num - prior_num_mu) ** 2) / (2 * aud_ind_var_num)
        ) / np.sqrt(2 * np.pi * aud_ind_var_num)
        likelihood_vis_num = np.exp(
            -((vis_num - prior_num_mu) ** 2) / (2 * vis_ind_var_num)
        ) / np.sqrt(2 * np.pi * vis_ind_var_num)

        # Joint likelihood for C=2 is the product of likelihoods
        likelihood_indep_num = likelihood_aud_num * likelihood_vis_num

        # Similar calculations for temporal dimension
        dis_common_time = (
            (aud_time - vis_time) ** 2 * prior_var_time
            + (aud_time - prior_time_mu) ** 2 * var_vis_time
            + (vis_time - prior_time_mu) ** 2 * var_aud_time
        )

        likelihood_common_time = np.exp(
            -dis_common_time / (2 * multisensory_var_time)
        ) / (2 * np.pi * np.sqrt(multisensory_var_time))

        likelihood_aud_time = np.exp(
            -((aud_time - prior_time_mu) ** 2) / (2 * aud_ind_var_time)
        ) / np.sqrt(2 * np.pi * aud_ind_var_time)
        likelihood_vis_time = np.exp(
            -((vis_time - prior_time_mu) ** 2) / (2 * vis_ind_var_time)
        ) / np.sqrt(2 * np.pi * vis_ind_var_time)

        likelihood_indep_time = likelihood_aud_time * likelihood_vis_time

        # Combine likelihoods from both dimensions, assuming independence
        likelihood_common_total = (
            likelihood_common_num * likelihood_common_time
        )

        # Calculate unnormalized posterior probability for common cause
        post_common = likelihood_common_total * p_common

        # Calculate normalization denominator
        # This factors the joint posterior over dimensions
        denom = (
            likelihood_common_num * p_common
            + likelihood_indep_num * (1 - p_common)
        ) * (
            likelihood_common_time * p_common
            + likelihood_indep_time * (1 - p_common)
        )

        # Normalized posterior probability of common cause
        pc = post_common / denom

        # Optimal combination of auditory, visual, and prior information in C=1
        common_est_num = (
            aud_num / var_aud_num
            + vis_num / var_vis_num
            + prior_num_mu / prior_var_num
        ) / (1 / var_aud_num + 1 / var_vis_num + 1 / prior_var_num)

        common_est_time = (
            aud_time / var_aud_time
            + vis_time / var_vis_time
            + prior_time_mu / prior_var_time
        ) / (1 / var_aud_time + 1 / var_vis_time + 1 / prior_var_time)

        # Different ways to combine C=1 and C=2 estimates based on pc
        if strategy == "selection":
            # Binary decision: use common-cause estimate if pc > 0.5
            final_aud_num = (pc > 0.5) * common_est_num + (
                pc <= 0.5
            ) * est_aud_num_ind
            final_vis_num = (pc > 0.5) * common_est_num + (
                pc <= 0.5
            ) * est_vis_num_ind
            final_aud_time = (pc > 0.5) * common_est_time + (
                pc <= 0.5
            ) * est_aud_time_ind
            final_vis_time = (pc > 0.5) * common_est_time + (
                pc <= 0.5
            ) * est_vis_time_ind
        elif strategy == "averaging":
            # Continuous weighted average based on posterior probability
            final_aud_num = pc * common_est_num + (1 - pc) * est_aud_num_ind
            final_vis_num = pc * common_est_num + (1 - pc) * est_vis_num_ind
            final_aud_time = pc * common_est_time + (1 - pc) * est_aud_time_ind
            final_vis_time = pc * common_est_time + (1 - pc) * est_vis_time_ind
        elif strategy == "matching":
            # Probabilistic selection: randomly choose based on pc
            match = 1 - self.random.random(self.n)
            final_aud_num = (pc > match) * common_est_num + (
                pc <= match
            ) * est_aud_num_ind
            final_vis_num = (pc > match) * common_est_num + (
                pc <= match
            ) * est_vis_num_ind
            final_aud_time = (pc > match) * common_est_time + (
                pc <= match
            ) * est_aud_time_ind
            final_vis_time = (pc > match) * common_est_time + (
                pc <= match
            ) * est_vis_time_ind
        else:
            # Default to averaging if strategy not recognized
            final_aud_num = pc * common_est_num + (1 - pc) * est_aud_num_ind
            final_vis_num = pc * common_est_num + (1 - pc) * est_vis_num_ind
            final_aud_time = pc * common_est_time + (1 - pc) * est_aud_time_ind
            final_vis_time = pc * common_est_time + (1 - pc) * est_vis_time_ind

        # Convert continuous estimates to discrete probability distributions
        # For numerosity:
        step_num = self.numerosity_res
        edges_num = numerosity_possible - step_num / 2
        edges_num = np.append(edges_num, edges_num[-1] + step_num)
        aud_num_hist = np.histogram(final_aud_num, edges_num)[0]
        vis_num_hist = np.histogram(final_vis_num, edges_num)[0]
        pred_aud_num = aud_num_hist / np.sum(aud_num_hist)
        pred_vis_num = vis_num_hist / np.sum(vis_num_hist)

        # For time:
        step_time = self.time_res
        edges_time = time_possible - step_time / 2
        edges_time = np.append(edges_time, edges_time[-1] + step_time)
        aud_time_hist = np.histogram(final_aud_time, edges_time)[0]
        vis_time_hist = np.histogram(final_vis_time, edges_time)[0]
        multi_time_hist = np.histogram(common_est_time, edges_time)[0]
        pred_aud_time = aud_time_hist / np.sum(aud_time_hist, axis=0)
        pred_vis_time = vis_time_hist / np.sum(vis_time_hist, axis=0)
        pred_multi_time = multi_time_hist / np.sum(multi_time_hist, axis=0)

        # ----- Return results -----
        return {
            "auditory_numerosity_ind": est_aud_num_ind,
            "auditory_time_ind": est_aud_time_ind,
            "visual_numerosity_ind": est_vis_num_ind,
            "visual_time_ind": est_vis_time_ind,
            "auditory_numerosity": pred_aud_num,
            "visual_numerosity": pred_vis_num,
            "auditory_time": pred_aud_time,
            "visual_time": pred_vis_time,
            "multi_time": pred_multi_time,
            "pc": pc,
        }

    def run(
        self,
        *,
        auditory_numerosity=2,
        visual_numerosity=1,
        auditory_numerosity_sigma=0.12,
        visual_numerosity_sigma=0.48,
        auditory_time=558.5,
        visual_time=500.0,
        auditory_time_sigma=40.0,
        visual_time_sigma=60.0,
        p_common=0.6,
        prior_numerosity_sigma=1.0,
        prior_numerosity_mu=1.0,
        prior_time_sigma=100.0,
        prior_time_mu=500.0,
        strategy="averaging",
        noise=True,
        causes_kind="count",
    ):
        """
        Runs the Multidimensional BCI model simulation.

        This is the main entry point for running a complete simulation.
        It orchestrates the entire process from generating noisy sensory
        measurements to computing final perceptual estimates across
        both dimensions.

        Parameters
        ----------
        auditory_numerosity : float
            Observed numerosity for the auditory modality (i.e., beeps).
        visual_numerosity : float
            Observed numerosity for the visual modality (i.e., flashes).
        auditory_numerosity_sigma : float
            Sensory noise (SD) for auditory numerosity.
        visual_numerosity_sigma : float
            Sensory noise (SD) for visual numerosity.
        auditory_time : float
            Observed onset time for the auditory stimulus (in ms).
        visual_time : float
            Observed onset time for the visual stimulus (in ms).
        auditory_time_sigma : float
            Sensory noise (SD) for auditory time (in ms).
        visual_time_sigma : float
            Sensory noise (SD) for visual time (in ms).
        p_common : float, default=0.5
            Prior probability of a common cause.
        prior_numerosity_sigma : float, default=5.0
            Prior standard deviation for numerosity.
        prior_numerosity_mu : float, default=5.0
            Prior mean for numerosity.
        prior_time_sigma : float, default=100.0
            Prior standard deviation for time (in ms).
        prior_time_mu : float, default=500.0
            Prior mean for time (in ms).
        strategy : str, default="averaging"
            Decision strategy: "selection", "averaging", or "matching".
        noise : bool, default=True
            Whether to include stochastic noise in sensory measurements.
        causes_kind : str, default="count"
            Type of cause calculation ("count" or "prob").

        Returns
        -------
        tuple
            A tuple containing:
              - dict: Response with perceptual distributions for each modality.
              - dict: Extra information including
              mean probability of common cause and numerosity estimates.

        """
        # Create discrete grids for binning numerosity and time estimates
        num_possible = np.arange(
            self.numerosity_range[0],
            self.numerosity_range[1],
            self.numerosity_res,
        )
        time_possible = np.arange(
            self.time_range[0],
            self.time_range[1],
            self.time_res,
        )

        # Add sensory noise to the true stimulus values
        aud_num_input = self.input_computation(
            auditory_numerosity, auditory_numerosity_sigma**2, noise=noise
        )
        vis_num_input = self.input_computation(
            visual_numerosity, visual_numerosity_sigma**2, noise=noise
        )
        aud_time_input = self.input_computation(
            auditory_time, auditory_time_sigma**2, noise=noise
        )
        vis_time_input = self.input_computation(
            visual_time, visual_time_sigma**2, noise=noise
        )

        # Calculate optimal Bayesian estimates for each modality and dimension
        est_aud_num_ind = self.unisensory_estimator(
            aud_num_input,
            auditory_numerosity_sigma,
            prior_numerosity_mu,
            prior_numerosity_sigma,
        )
        est_vis_num_ind = self.unisensory_estimator(
            vis_num_input,
            visual_numerosity_sigma,
            prior_numerosity_mu,
            prior_numerosity_sigma,
        )
        est_aud_time_ind = self.unisensory_estimator(
            aud_time_input,
            auditory_time_sigma,
            prior_time_mu,
            prior_time_sigma,
        )
        est_vis_time_ind = self.unisensory_estimator(
            vis_time_input, visual_time_sigma, prior_time_mu, prior_time_sigma
        )

        # Run the full multidimensional causal inference model
        multisensory_est = self.multisensory_estimator(
            aud_num_input,
            vis_num_input,
            aud_time_input,
            vis_time_input,
            auditory_numerosity_sigma,
            visual_numerosity_sigma,
            auditory_time_sigma,
            visual_time_sigma,
            prior_numerosity_mu,
            prior_numerosity_sigma,
            prior_time_mu,
            prior_time_sigma,
            p_common,
            strategy,
            num_possible,
            time_possible,
            est_aud_num_ind,
            est_vis_num_ind,
            est_aud_time_ind,
            est_vis_time_ind,
        )

        # Prepare output arrays (2D: time x space)
        pos_range = np.arange(
            self.position_range[0], self.position_range[1], self.position_res
        )

        aud_time_res = np.zeros((time_possible.size, pos_range.size))
        vis_time_res = np.zeros((time_possible.size, pos_range.size))
        multi_time_res = np.zeros((time_possible.size, pos_range.size))

        # Assign the binned distributions
        aud_num_res = multisensory_est["auditory_numerosity"]
        vis_num_res = multisensory_est["visual_numerosity"]
        aud_time_res[:, 0] = multisensory_est["auditory_time"]
        vis_time_res[:, 0] = multisensory_est["visual_time"]
        multi_time_res[:, 0] = multisensory_est["multi_time"]

        # Compile responses and extras
        response = {
            "auditory": aud_time_res,
            "visual": vis_time_res,
            "multi": multi_time_res,
        }

        extra = {
            "mean_p_common_cause": np.average(multisensory_est["pc"]),
            "p_common_cause": multisensory_est["pc"],
            "causes_kind": causes_kind,
            "auditory_numerosity": aud_num_res,
            "visual_numerosity": vis_num_res,
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
