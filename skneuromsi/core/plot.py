#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the Result object."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Iterable

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from ..utils import AccessorABC

# =============================================================================
# HELPERS
# =============================================================================


def max_value(dim, xa):
    arr = xa[dim].to_numpy()
    idx = xa.argmax(...)[dim].to_numpy()
    return arr[idx]


def mean_max_value(dim, xa):
    arr = xa[dim].to_numpy()
    groups = xa.groupby(dim)
    groups_cvar = groups.mean(...)
    idx = groups_cvar.argmax(...)[dim].to_numpy()
    return arr[idx]


HEURISTICS = {
    "max_value": max_value,
    "mean_max_value": mean_max_value,
}

# =============================================================================
# PLOTTER OBJECT
# =============================================================================


class ResultPlotter(AccessorABC):
    """Make plots of Result.

    Kind of plot to produce:



    """

    _default_kind = "pline"

    def __init__(self, result):
        self._result = result

    # LINE ====================================================================
    def _resolve_axis(self, positions_coordinates_number, ax):

        if ax is None:
            fig, ax = plt.subplots(
                1, positions_coordinates_number, sharey=True
            )

            size_x, size_y = fig.get_size_inches()
            fig.set_size_inches(size_x * positions_coordinates_number, size_y)

        if not isinstance(ax, Iterable):
            ax = np.array([ax])

        return ax

    def pline(self, time="mean_max_value", **kwargs):

        coords = self._result.pcoords_

        ax = self._resolve_axis(len(coords), kwargs.pop("ax", None))

        xa = self._result.to_xarray()

        if time in HEURISTICS:
            timeheu = HEURISTICS[time]
            time = timeheu("times", xa)

        for coord, ax in zip(coords, ax):

            df = xa.sel(positions_coordinates=coord, times=time).to_dataframe()

            sns.lineplot(
                x="positions",
                y="values",
                hue="modes",
                data=df,
                ax=ax,
                **kwargs,
            )

        ax.set_title(f"Time {time}")
        ax.legend()

        return ax

    def tlinel(self, position="mean_max_value", **kwargs):

        coords = self._result.pcoords_

        ax = self._resolve_axis(len(coords), kwargs.pop("ax", None))

        xa = self._result.to_xarray()

        if position in HEURISTICS:
            posheu = HEURISTICS[position]
            position = posheu("positions", xa)

        for coord, ax in zip(coords, ax):

            df = xa.sel(
                positions_coordinates=coord, positions=position
            ).to_dataframe()

            sns.lineplot(
                x="times", y="values", hue="modes", data=df, ax=ax, **kwargs
            )

        ax.set_title(f"Position {position}")
        ax.legend()

        return ax
