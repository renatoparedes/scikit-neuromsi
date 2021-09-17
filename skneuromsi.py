import abc

import attr
import matplotlib.pyplot as plt
import numpy as np


class MSIBrain(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __len__(self):
        """Number of modalities."""

    @abc.abstractmethod
    def __getitem__(self, modality):
        """"""

    @abc.abstractmethod
    def response(self):
        """"""


@attr.s
class AlaisBurr2004(MSIBrain):

    # Stimuli locations
    locs = attr.ib(default={"auditory": 0, "visual": 0})

    # All possible locations
    pos_locs = attr.ib(default=np.arange(-20, 20, 0.01))

    # Weights used to calculate the mean of the multimodal distribution
    weights = attr.ib(default={"auditory": 0.5, "visual": 0.5})

    # Variability of stimulus estimates
    e_sigmas = attr.ib(default={"auditory": 3, "visual": 4})

    # Result
    _result = attr.ib(factory=dict)

    def __attrs_post_init__(self):

        # From Alais and Burr 2004, the weights used to calculation
        # the mean of the multimodal distribution are:

        self.weights["auditory"] = self.e_sigmas["visual"] ** 2 / (
            self.e_sigmas["auditory"] ** 2 + self.e_sigmas["visual"] ** 2
        )

        self.weights["visual"] = self.e_sigmas["auditory"] ** 2 / (
            self.e_sigmas["visual"] ** 2 + self.e_sigmas["auditory"] ** 2
        )

        # From both Alais/Burr, 2004 and Ernst/Banks, 2002,
        # the multisensory variability is:
        self.e_sigmas["multisensory"] = np.sqrt(
            (self.e_sigmas["visual"] ** 2 * self.e_sigmas["auditory"] ** 2)
            / (self.e_sigmas["auditory"] ** 2 + self.e_sigmas["visual"] ** 2)
        )

        # And the multisensory loc is:
        self.locs["multisensory"] = (
            self.weights["visual"] * self.locs["visual"]
            + self.weights["auditory"] * self.locs["auditory"]
        )

    def auditory_modality(self):
        """
        Computes auditory estimate
        """

        if "auditiva" in self._result:
            raise ValueError()

        distr_a = (
            1 / np.sqrt(2 * np.pi * self.e_sigmas["auditory"] ** 2)
        ) * np.exp(
            -1
            * (
                ((self.pos_locs - self.locs["auditory"]) ** 2)
                / (2 * self.e_sigmas["auditory"] ** 2)
            )
        )

        self._result["auditory"] = distr_a

    def visual_modality(self):
        """
        Computes visual estimate
        """

        if "visual" in self._result:
            raise ValueError()

        distr_v = (
            1 / np.sqrt(2 * np.pi * self.e_sigmas["visual"] ** 2)
        ) * np.exp(
            -1
            * (
                ((self.pos_locs - self.locs["visual"]) ** 2)
                / (2 * self.e_sigmas["visual"] ** 2)
            )
        )

        self._result["visual"] = distr_v

    def multisensory_modality(self):
        """
        Computes multisensory estimate
        """

        if "multisensory" in self._result:
            raise ValueError()

        distr_m = (
            1 / np.sqrt(2 * np.pi * self.e_sigmas["multisensory"] ** 2)
        ) * np.exp(
            -1
            * (
                ((self.pos_locs - self.locs["multisensory"]) ** 2)
                / (2 * self.e_sigmas["multisensory"] ** 2)
            )
        )

        self._result["multisensory"] = distr_m

    def __len__(self):
        return 2

    def __getitem__(self, modality):
        if modality == "auditory":
            return self.auditory_modality
        elif modality == "visual":
            return self.visual_modality
        elif modality == "multisensory":
            return self.visual_modality
        raise KeyError(modality)

    def response(self):
        return self._result["multisensory"]

    def plot(self):
        for res in self._result:
            plt.plot(self.pos_locs, self._result[res])
        plt.ylabel("Probability density", size=12)
        plt.xlabel("Position (degrees of visual angle)", size=12)
        plt.legend(self._result.keys())

    def run(self):
        self.auditory_modality()
        self.visual_modality()
        self.multisensory_modality()


AlaisBurr2004()
