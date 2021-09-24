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
    locs = attr.ib(factory=dict)

    # All possible locations - Experiment setup
    pos_locs = attr.ib()

    @pos_locs.default
    def _pos_locs_default(self):
        return np.arange(-20, 20, 0.01)

    # Weights used to calculate the mean of the multimodal distribution
    weights = attr.ib(factory=dict)

    # Variability of stimulus estimates
    e_sigmas = attr.ib(factory=dict)

    # Result
    _result = attr.ib(factory=dict)

    def __attrs_post_init__(self):

        # Stimuli setup
        self.locs["auditory"] = 0
        self.locs["visual"] = 0

        loc_a = self.locs["auditory"]
        loc_v = self.locs["visual"]

        # Model setup
        self.e_sigmas["auditory"] = 3
        self.e_sigmas["visual"] = 4

        sig_v = self.e_sigmas["auditory"]
        sig_a = self.e_sigmas["visual"]

        # From Alais and Burr 2004, the weights used to calculation
        # the mean of the multimodal distribution are:

        self.weights["auditory"] = sig_v ** 2 / (sig_a ** 2 + sig_v ** 2)

        self.weights["visual"] = sig_a ** 2 / (sig_v ** 2 + sig_a ** 2)

        w_a = self.weights["auditory"]
        w_v = self.weights["visual"]

        # From both Alais/Burr, 2004 and Ernst/Banks, 2002,
        # the multisensory variability is:
        self.e_sigmas["multisensory"] = np.sqrt(
            (sig_v ** 2 * sig_a ** 2) / (sig_a ** 2 + sig_v ** 2)
        )

        # And the multisensory loc is:
        self.locs["multisensory"] = w_v * loc_v + w_a * loc_a

    def auditory_modality(self):
        """
        Computes auditory estimate
        """

        sig_a = self.e_sigmas["auditory"]
        loc_a = self.locs["auditory"]
        locs = self.pos_locs

        if "auditory" in self._result:
            raise ValueError()

        distr_a = (1 / np.sqrt(2 * np.pi * sig_a ** 2)) * np.exp(
            -1 * (((locs - loc_a) ** 2) / (2 * sig_a ** 2))
        )

        self._result["auditory"] = distr_a

    def visual_modality(self):
        """
        Computes visual estimate
        """

        sig_v = self.e_sigmas["visual"]
        loc_v = self.locs["visual"]
        locs = self.pos_locs

        if "visual" in self._result:
            raise ValueError()

        distr_v = (1 / np.sqrt(2 * np.pi * sig_v ** 2)) * np.exp(
            -1 * (((locs - loc_v) ** 2) / (2 * sig_v ** 2))
        )

        self._result["visual"] = distr_v

    def multisensory_modality(self):
        """
        Computes multisensory estimate
        """
        sig_m = self.e_sigmas["multisensory"]
        loc_m = self.locs["multisensory"]
        locs = self.pos_locs

        if "multisensory" in self._result:
            raise ValueError()

        distr_m = (1 / np.sqrt(2 * np.pi * sig_m ** 2)) * np.exp(
            -1 * (((locs - loc_m) ** 2) / (2 * sig_m ** 2))
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
