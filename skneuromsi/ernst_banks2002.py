# https://www.nature.com/articles/415429a

import numpy as np

from . import core


def haptic_estimator(haptic_sigma, posible_heights, haptic_height):
    sigma = haptic_sigma
    height = haptic_height
    pheights = posible_heights

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )


def visual_estimator(posible_heights, visual_sigma, visual_height):
    pheights = posible_heights
    height = visual_height
    sigma = visual_sigma

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )


def multisensory_estimator(
    multisensory_sigma,
    posible_heights,
    visual_height,
    haptic_height,
    visual_weight,
    haptic_weight,
):
    """
    Computes multisensory estimate
    """

    sigma = multisensory_sigma

    height = visual_weight * visual_height + haptic_weight * haptic_height
    pheights = posible_heights

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((pheights - height) ** 2) / (2 * sigma ** 2))
    )


@core.neural_msi_model
class ErnstBanks2002:

    # hiper parameters
    posible_heights = core.hparameter(factory=lambda: np.arange(49, 61, 0.01))
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

    # estimulii!
    stimuli = [haptic_estimator, visual_estimator]
    integration = multisensory_estimator
