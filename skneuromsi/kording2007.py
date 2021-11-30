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


def auditory_estimator(
    auditory_location, auditory_var, auditory_var_hat, prior_var, prior_mu, N
):
    """Computes the auditory estimate.

    Parameters
    ----------
    auditory_location: ``float``
        Location in which the auditory stimulus is delivered.
    auditory_var: ``float``
        Variance of the auditory signal.
    auditory_var_hat: ``float``
        Estimated variance of the auditory signal.
    prior_var: ``float``
        Variance of the prior distribuion of the stimulus location.
    prior_mu: ``float``
        Mean of the prior distribuion of the stimulus location.
    N: ``float``
        Number of Monte Carlo samples to take in computing the model estimates.


    Returns
    ----------
    auditory_estimate: ``dict``
        Results of the auditory unimodal estimator.
        Includes these fields:

        * *"auditory_input"*: Numpy array containing the received
            auditory stimuli
        * *"auditory_hat_ind"*: Numpy array containing the estimated
            auditory location for an independent cause.
    """

    auditory_input = auditory_location + np.sqrt(
        auditory_var
    ) * np.random.randn(N)

    auditory_hat_ind = (
        (auditory_input / auditory_var) + (np.ones(N) * prior_mu) / prior_var
    ) * auditory_var_hat

    auditory_estimate = {
        "auditory_input": auditory_input,
        "auditory_hat_ind": auditory_hat_ind,
    }

    return auditory_estimate


def visual_estimator(
    visual_location, visual_var, visual_var_hat, prior_var, prior_mu, N
):
    """Computes the visual estimate.

    Parameters
    ----------
    visual_location: ``float``
        Location in which the visual stimulus is delivered.
    visual_var: ``float``
        Variance of the visual signal.
    visual_var_hat: ``float``
        Estimated variance of the visual signal.
    prior_var: ``float``
        Variance of the prior distribuion of the stimulus location.
    prior_mu: ``float``
        Mean of the prior distribuion of the stimulus location.
    N: ``float``
        Number of Monte Carlo samples to take in computing the model estimates.


    Returns
    ----------
    visual_estimate: ``dict``
        Results of the visual unimodal estimator.
        Includes these fields:

        * *"visual_input"*: Numpy array containing the received
            visual stimuli
        * *"visual_hat_ind"*: Numpy array containing the estimated visual
            location for an independent cause.
    """

    visual_input = visual_location + np.sqrt(visual_var) * np.random.randn(N)

    visual_hat_ind = (
        (visual_input / visual_var) + (np.ones(N) * prior_mu) / prior_var
    ) * visual_var_hat

    visual_estimate = {
        "visual_input": visual_input,
        "visual_hat_ind": visual_hat_ind,
    }

    return visual_estimate


def multisensory_estimator(
    auditory_estimator,
    visual_estimator,
    auditory_var,
    visual_var,
    prior_var,
    prior_mu,
    multisensory_var,
    multisensory_var_hat,
    N,
    auditory_ind_var,
    visual_ind_var,
    p_common,
    strategy,
    possible_locations,
):

    """
    Computes the multisensory estimate.

    Parameters
    ----------
    auditory_estimator: ``dict``
        Results of the auditory unimodal estimator.
        Includes these fields:

        * *"auditory_input"*: Numpy array containing the received
            auditory stimuli
        * *"auditory_hat_ind"*: Numpy array containing the estimated
            auditory location for an independent cause.
    visual_estimator: ``dict``
        Results of the visual unimodal estimator.
        Includes these fields:

        * *"visual_input"*: Numpy array containing the received
            visual stimuli
        * *"visual_hat_ind"*: Numpy array containing the estimated visual
            location for an independent cause.
    auditory_var: ``float``
        Variance of the auditory signal.
    visual_var: ``float``
        Variance of the visual signal.
    prior_var: ``float``
        Variance of the prior distribuion of the stimulus location.
    prior_mu: ``float``
        Mean of the prior distribuion of the stimulus location.
    multisensory_var: ``float``
        Variance of the audio-visual signal.
    multisensory_var_hat: ``float``
        Estimated variance of the audio-visual signal.
    N: ``float``
        Number of Monte Carlo samples to take in computing the model estimates.
    auditory_ind_var: ``float``
        Variance of the auditory signal for an independent cause.
    visual_ind_var: ``float``
        Variance of the visual signal for an independent cause.
    p_common: ``float``
        Probability of signals coming from a single cause, i.e. P(C=1).
    strategy: ``str``
        Decision strategy for producing estimates of stimulus location.
        Accepts "selection", "averaging" and "mapping".
    possible_locations: ``ndarray``
        Numpy array containing all the possible locations where the stimuli
        could be delivered.

    Returns
    ----------
    res: ``dict``
        Results of the multisensory integration BCI model.
        Includes these fields:

        * *"auditory"*: Auditory location estimate.
        * *"visual"*: Visual location estimate.

    """

    # Inputs
    auditory_input = auditory_estimator["auditory_input"]
    visual_input = visual_estimator["visual_input"]
    single_stim = np.sum(np.isnan([auditory_input, visual_input]))

    # Perceived location of causes
    auditory_hat_ind = auditory_estimator["auditory_hat_ind"]
    visual_hat_ind = visual_estimator["visual_hat_ind"]
    multisensory_hat = (
        (auditory_input / auditory_var)
        + (visual_input / visual_var)
        + (np.ones(N) * prior_mu) / prior_var
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
    likelihood_visual = np.exp(-dis_visual / (2 * visual_ind_var)) / np.sqrt(
        2 * np.pi * visual_ind_var
    )
    likelihood_indep = likelihood_auditory * likelihood_visual
    post_common = likelihood_common * p_common
    post_indep = likelihood_indep * (1 - p_common)
    pC = post_common / (post_common + post_indep)

    # Independent Causes
    if single_stim:
        auditory_hat = auditory_hat_ind
        visual_hat = visual_hat_ind
    else:
        # Decision Strategies
        # Model Selection
        if strategy == "selection":
            auditory_hat = (pC > 0.5) * multisensory_hat + (
                pC <= 0.5
            ) * auditory_hat_ind
            visual_hat = (pC > 0.5) * multisensory_hat + (
                pC <= 0.5
            ) * visual_hat_ind
        # Model Averaging
        elif strategy == "averaging":
            auditory_hat = (pC) * multisensory_hat + (
                1 - pC
            ) * auditory_hat_ind
            visual_hat = (pC) * multisensory_hat + (1 - pC) * visual_hat_ind
        # Model Matching
        elif strategy == "matching":
            match = 1 - np.random.rand(N)
            auditory_hat = (pC > match) * multisensory_hat + (
                pC <= match
            ) * auditory_hat_ind
            visual_hat = (pC > match) * multisensory_hat + (
                pC <= match
            ) * visual_hat_ind

    # Prediction of location estimates
    step = possible_locations[1]
    edges = possible_locations[0] - step / 2
    edges = np.append(edges, edges[-1] + step)

    auditory_estimates = np.histogram(auditory_hat, edges)[0]
    visual_estimates = np.histogram(visual_hat, edges)[0]

    pred_auditory = auditory_estimates / np.sum(auditory_estimates, axis=0)
    pred_visual = visual_estimates / np.sum(visual_estimates, axis=0)

    res = {"auditory": pred_auditory, "visual": pred_visual}

    return res


# ===============================================================================
# CLASSES
# ===============================================================================


@core.neural_msi_model
class Kording2007:
    """Class that implements the Bayesian Causal Inference model for
    Multisensory Perception employed by Kording et al. to reproduce
    the Ventriloquist Effect [3]_. The class follows the implementation
    provided in the Bayesian Causal Inferente Toolbox (BCIT) [4]_.

    Attributes
    ----------
    possible_locations: ``skneuromsi.hparameter``
        All the possible locations where the stimulus
        could be delivered.
    auditory_sigma: ``skneuromsi.hparameter``
        Standard deviation of the auditory estimate.
    auditory_var: ``skneuromsi.hparameter``
        Variance of the auditory estimate.
    visual_sigma: ``skneuromsi.hparameter``
        Standard deviation of the visual estimate.
    visual_var: ``skneuromsi.hparameter``
        Variance of the visual estimate.
    p_common: ``skneuromsi.internal``
        Probability of signals coming from a single cause, i.e. P(C=1).
    prior_sigma: ``skneuromsi.internal``
        Standard deviation of the prior distribuion of the stimulus location.
    prior_var: ``skneuromsi.internal``
        Variance of the prior distribuion of the stimulus location.
    strategy: ``skneuromsi.internal``
        Decision strategy for producing estimates of stimulus location.
        Accepts "selection", "averaging" and "mapping".
    prior_mu: ``skneuromsi.internal``
        Mean of the prior distribuion of the stimulus location.
    multisensory_var: ``skneuromsi.internal``
        Variance of the audio-visual signal.
    auditory_ind_var: ``skneuromsi.internal``
        Variance of the auditory signal for an independent cause.
    visual_ind_var: ``skneuromsi.internal``
        Variance of the visual signal for an independent cause.
    multisensory_var_hat: ``skneuromsi.internal``
        Estimated variance of the audio-visual signal.
    auditory_var_hat: ``skneuromsi.internal``
        Estimated variance of the auditory signal.
    visual_var_hat: ``skneuromsi.internal``
        Estimated variance of the visual signal.
    stimuli: ``list`` of ``callable``
        List containing the functions employed for the
        computation of unisensory estimates.
    integration: ``callable``
        Function to compute the multisensory estimate.

    References
    ----------
    .. [3] K. P. Körding, U. Beierholm, W. J. Ma, S. Quartz, J. B. Tenenbaum,
        and L. Shams, “Causal Inference in Multisensory Perception,” PLoS One,
        vol. 2, no. 9, p. e943, Sep. 2007, doi: 10.1371/journal.pone.0000943.
    .. [4] M. Samad, K. Sita, A. Wang and L. Shams. Bayesian Causal Inference
        Toolbox (BCIT) for MATLAB.
        https://shamslab.psych.ucla.edu/bci-matlab-toolbox/
    """

    # hiperparameters
    possible_locations = core.hparameter(
        factory=lambda: np.linspace(-42, 42, 50, retstep=True)
    )

    N = core.hparameter(default=10000)

    auditory_sigma = core.hparameter(default=2)
    auditory_var = core.hparameter()

    @auditory_var.default
    def _auditory_var_default(self):
        return self.auditory_sigma ** 2

    visual_sigma = core.hparameter(default=10)
    visual_var = core.hparameter()

    @visual_var.default
    def _visual_var_default(self):
        return self.visual_sigma ** 2

    # internals
    p_common = core.internal(default=0.5)
    prior_sigma = core.internal(default=20)
    prior_var = core.internal()

    strategy = core.internal(default="averaging")

    @prior_var.default
    def _prior_var_default(self):
        return self.prior_sigma ** 2

    prior_mu = core.internal(default=0)

    multisensory_var = core.internal()

    @multisensory_var.default
    def _multisensory_var_default(self):
        return (
            self.auditory_var * self.visual_var
            + self.auditory_var * self.prior_var
            + self.visual_var * self.prior_var
        )

    auditory_ind_var = core.internal()

    @auditory_ind_var.default
    def _auditory_ind_var_default(self):
        return self.auditory_var + self.prior_var

    visual_ind_var = core.internal()

    @visual_ind_var.default
    def _visual_ind_var_default(self):
        return self.visual_var + self.prior_var

    multisensory_var_hat = core.internal()

    @multisensory_var_hat.default
    def _multisensory_var_hat_default(self):
        return 1 / (
            1 / self.auditory_var + 1 / self.visual_var + 1 / self.prior_var
        )

    auditory_var_hat = core.internal()

    @auditory_var_hat.default
    def _auditory_var_hat_default(self):
        return 1 / (1 / self.auditory_var + 1 / self.prior_var)

    visual_var_hat = core.internal()

    @visual_var_hat.default
    def _visual_var_hat_default(self):
        return 1 / (1 / self.visual_var + 1 / self.prior_var)

    # estimators
    stimuli = [auditory_estimator, visual_estimator]
    integration = multisensory_estimator
