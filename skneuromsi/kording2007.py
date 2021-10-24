# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943
# https://github.com/multisensoryperceptionlab/BCIT
# https://github.com/multisensoryperceptionlab/BCIT/blob/master/BCIM%20ToolBox/bciModel.m

import numpy as np

from . import core


def auditory_estimator(
    auditory_input, auditory_var, auditory_var_hat, prior_var, N, prior_mu
):
    return (
        (auditory_input / auditory_var) + (np.ones(N) * prior_mu) / prior_var
    ) * auditory_var_hat


def visual_estimator(
    visual_input, visual_var, visual_var_hat, prior_var, N, prior_mu
):
    return (
        (visual_input / visual_var) + (np.ones(N) * prior_mu) / prior_var
    ) * visual_var_hat


def multisensory_estimator(
    auditory_estimator,
    visual_estimator,
    auditory_input,
    auditory_var,
    visual_input,
    visual_var,
    prior_var,
    prior_mu,
    multisensory_var_hat,
    N,
    dis_common,
    dis_visual,
    dis_auditory,
    multisensory_com_var,
    auditory_ind_var,
    visual_ind_var,
    p_common,
    single_stim,
    strategy,
):

    # Perceived location of causes
    multisensory_estimator_common = (
        (auditory_input / auditory_var)
        + (visual_input / visual_var)
        + (np.ones(N) * prior_mu) / prior_var
    ) * multisensory_var_hat

    # Likelihood calculations
    likelihood_common = np.exp(-dis_common / (2 * multisensory_com_var)) / (
        2 * np.pi * np.sqrt(multisensory_com_var)
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
        s1_hat = auditory_estimator
        s2_hat = visual_estimator
    else:
        # Decision Strategies
        # Model Selection
        if strategy == "Selection":
            s1_hat = (pC > 0.5) * multisensory_estimator_common + (
                pC <= 0.5
            ) * auditory_estimator
            s2_hat = (pC > 0.5) * multisensory_estimator_common + (
                pC <= 0.5
            ) * visual_estimator
        # Model Matching
        elif strategy == "Matching":
            s1_hat = (pC) * multisensory_estimator_common + (
                1 - pC
            ) * auditory_estimator
            s2_hat = (pC) * multisensory_estimator_common + (
                1 - pC
            ) * visual_estimator
        # Model Averaging
        elif strategy == "Averaging":
            match = 1 - np.random.rand(N)
            s1_hat = (pC > match) * multisensory_estimator_common + (
                pC <= match
            ) * auditory_estimator
            s2_hat = (pC > match) * multisensory_estimator_common + (
                pC <= match
            ) * visual_estimator

    # Prediction of location estimates
    # h1 = np.hist(s1_hat, space)
    # h2 = np.hist(s2_hat, space)

    # pred1 = bsxfun(@rdivide,h1,np.sum(h1))
    # pred2 = bsxfun(@rdivide,h2,np.sum(h2))

    pred1 = s1_hat
    pred2 = s2_hat

    return pred1, pred2


@core.neural_msi_model
class Kording2007:

    # hiperparameters
    # TODO look up for default values
    # 1 =  auditory, 2 = visual

    auditory_sigma = core.hparameter()
    auditory_var = core.hparameter()

    @auditory_var.default
    def _auditory_var_default(self):
        return self.auditory_var ** 2

    visual_sigma = core.hparameter()
    visual_var = core.hparameter()

    @visual_var.default
    def _visual_var_default(self):
        return self.visual_var ** 2

    N = core.hparameter(default=2)  # TODO double check.

    # Inputs
    auditory_location = core.hparameter()
    visual_location = core.hparameter()

    auditory_input = core.hparameter()

    @auditory_input.default
    def _auditory_input_default(self):
        return self.auditory_location + self.auditory_sigma * np.random.randn(
            self.N
        )

    visual_input = core.hparameter()

    @visual_input.default
    def _visual_input_default(self):
        return self.visual_location + self.visual_sigma * np.random.randn(
            self.N
        )

    dis_common = core.hparameter()

    @dis_common.default
    def _dis_common_default(self):
        return (
            (self.auditory_input - self.visual_input) ** 2 * self.prior_var
            + (self.auditory_input - self.prior_mu) ** 2 * self.visual_var
            + (self.visual_input - self.prior_mu) ** 2 * self.auditory_var
        )

    dis_auditory = core.hparameter()

    @dis_auditory.default
    def _dis_auditory_default(self):
        return (self.auditory_input - self.prior_mu) ** 2

    dis_visual = core.hparameter()

    @dis_visual.default
    def _dis_visual_default(self):
        return (self.visual_input - self.prior_mu) ** 2

    single_stim = core.hparameter()

    @single_stim.default
    def _single_stim_default(self):
        return np.sum(np.isnan([self.auditory_location, self.visual_location]))

    # internals

    p_common = core.internal(default=0.5)
    prior_sigma = core.internal()
    prior_var = core.internal()

    strategy = core.internal(default="Averaging")

    @prior_var.default
    def _prior_var_default(self):
        return self.prior_sigma ** 2

    prior_mu = core.internal()

    multisensory_com_var = core.internal()

    @multisensory_com_var.default
    def _multisensory_com_var_default(self):
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
