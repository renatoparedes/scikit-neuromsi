import ipdb
import numpy as np

import skneuromsi as sknm


def auditory_stimulus(auditory_sigma, posible_locations, auditory_location):
    sigma = auditory_sigma
    location = auditory_location
    plocations = posible_locations

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


def visual_stimulus(posible_locations, visual_sigma, visual_location):
    plocations = posible_locations
    location = visual_location
    sigma = visual_sigma

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


def multisensory_stimulus(
    multisensory_sigma,
    posible_locations,
    visual_stimulus,
    auditory_stimulus,
):
    """
    Computes multisensory estimate
    """
    sigma = multisensory_sigma
    location = 2+2 #multisensory_location
    plocations = posible_locations

    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (((plocations - location) ** 2) / (2 * sigma ** 2))
    )


@sknm.neural_msi_model
class AlaisBurr2004:

    # hiper parameters
    posible_locations = sknm.hparameter(
        factory=lambda: np.arange(-20, 20, 0.01)
    )
    auditory_sigma = sknm.hparameter(default=3.0)
    visual_sigma = sknm.hparameter(default=4.0)

    # internals
    auditory_weight = sknm.internal(default=1)
    visual_weight = sknm.internal(default=2)
    multisensory_sigma = sknm.internal(default=3)


    # estimulii!
    stimuli = [auditory_stimulus, visual_stimulus]
    integration = multisensory_stimulus


model = AlaisBurr2004()
model.run(visual_location=1, auditory_location=3)

