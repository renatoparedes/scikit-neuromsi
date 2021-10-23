import numpy as np

from . import core


def auditory_estimator(auditory_sigma, posible_locations, auditory_location):
    sigma = auditory_sigma
    location = auditory_location
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
    posible_locations,
    visual_location,
    auditory_location,
    visual_weight,
    auditory_weight,
    multisensory_sigma,
):
    """
    Computes multisensory estimate
    """

    sigma = multisensory_sigma

    location = (
        visual_weight * visual_location + auditory_weight * auditory_location
    )
    plocations = posible_locations

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


@core.neural_msi_model
class AlaisBurr2004:

    # hiper parameters
    posible_locations = core.hparameter(
        factory=lambda: np.arange(-20, 20, 0.01)
    )
    auditory_sigma = core.hparameter(default=3.0)
    visual_sigma = core.hparameter(default=3.0)

    # internals
    auditory_weight = core.internal()

    @auditory_weight.default
    def _auditory_weight_default(self):
        return self.visual_sigma ** 2 / (
            self.auditory_sigma ** 2 + self.visual_sigma ** 2
        )

    visual_weight = core.internal()

    @visual_weight.default
    def _visual_weights_default(self):
        return self.auditory_sigma ** 2 / (
            self.visual_sigma ** 2 + self.auditory_sigma ** 2
        )

    multisensory_sigma = core.internal()

    @multisensory_sigma.default
    def _multisensory_sigma_default(self):
        return np.sqrt(
            (self.visual_sigma ** 2 * self.auditory_sigma ** 2)
            / (self.auditory_sigma ** 2 + self.visual_sigma ** 2)
        )

    # estimulii!
    stimuli = [auditory_estimator, visual_estimator]
    integration = multisensory_estimator
