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
