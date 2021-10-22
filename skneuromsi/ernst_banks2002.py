# https://www.nature.com/articles/415429a

import numpy as np

from . import core


def haptic_estimator(haptic_sigma, posible_locations, haptic_location):
    sigma = haptic_sigma
    location = haptic_location
    plocations = posible_locations

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


def visual_estimator(posible_locations, visual_sigma, visual_location):
    plocations = posible_locations
    location = visual_location
    sigma = visual_sigma

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


def multisensory_estimator(
    multisensory_sigma,
    posible_locations,
    visual_location,
    haptic_location,
    visual_weight,
    haptic_weight,
):
    """
    Computes multisensory estimate
    """

    sigma = multisensory_sigma

    location = (
        visual_weight * visual_location + haptic_weight * haptic_location
    )
    plocations = posible_locations

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


@core.neural_msi_model
class ErnstBanks2002:

    # hiper parameters
    posible_locations = core.hparameter(
        factory=lambda: np.arange(-20, 20, 0.01) # TODO fix locations
    )
    haptic_sigma = core.hparameter(default=4)
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
