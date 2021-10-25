# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943
# https://github.com/multisensoryperceptionlab/BCIT
# https://github.com/multisensoryperceptionlab/BCIT/blob/master/BCIM%20ToolBox/bciModel.m

# As implemented in the BCIM Toolbox

import numpy as np

from . import core


def auditory_estimator(
    auditory_location, auditory_var, auditory_var_hat, prior_var, N, prior_mu
):
    auditory_input = auditory_location + np.sqrt(
        auditory_var
    ) * np.random.randn(N)

    auditory_hat_ind = (
        (auditory_input / auditory_var) + (np.ones(N) * prior_mu) / prior_var
    ) * auditory_var_hat

    return {
        "auditory_input": auditory_input,
        "auditory_hat_ind": auditory_hat_ind,
    }


def visual_estimator(
    visual_location, visual_var, visual_var_hat, prior_var, N, prior_mu
):

    visual_input = visual_location + np.sqrt(visual_var) * np.random.randn(N)

    visual_hat_ind = (
        (visual_input / visual_var) + (np.ones(N) * prior_mu) / prior_var
    ) * visual_var_hat

    return {"visual_input": visual_input, "visual_hat_ind": visual_hat_ind}


def multisensory_estimator(
    auditory_estimator,
    visual_estimator,
    auditory_var,
    visual_var,
    prior_var,
    prior_mu,
    multisensory_var_hat,
    N,
    multisensory_var,
    auditory_ind_var,
    visual_ind_var,
    p_common,
    strategy,
    posible_locations,
):
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
        if strategy == "Selection":
            auditory_hat = (pC > 0.5) * multisensory_hat + (
                pC <= 0.5
            ) * auditory_hat_ind
            visual_hat = (pC > 0.5) * multisensory_hat + (
                pC <= 0.5
            ) * visual_hat_ind
        # Model Averaging
        elif strategy == "Averaging":
            auditory_hat = (pC) * multisensory_hat + (
                1 - pC
            ) * auditory_hat_ind
            visual_hat = (pC) * multisensory_hat + (1 - pC) * visual_hat_ind
        # Model Matching
        elif strategy == "Matching":
            match = 1 - np.random.rand(N)
            auditory_hat = (pC > match) * multisensory_hat + (
                pC <= match
            ) * auditory_hat_ind
            visual_hat = (pC > match) * multisensory_hat + (
                pC <= match
            ) * visual_hat_ind

    # Prediction of location estimates
    step = posible_locations[1]
    edges = posible_locations[0] - step / 2
    edges = np.append(edges, edges[-1] + step)

    auditory_estimates = np.histogram(auditory_hat, edges)[0]
    visual_estimates = np.histogram(visual_hat, edges)[0]

    pred_auditory = auditory_estimates / np.sum(auditory_estimates, axis=0)
    pred_visual = visual_estimates / np.sum(visual_estimates, axis=0)

    return {"auditory": pred_auditory, "visual": pred_visual}


@core.neural_msi_model
class Kording2007:

    # hiperparameters
    posible_locations = core.hparameter(
        factory=lambda: np.linspace(-42, 42, 50, retstep=True)
    )

    N = core.hparameter(default=100000)

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

    strategy = core.internal(default="Averaging")

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
