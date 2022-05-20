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

import xarray as xr

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

    _default_kind = "linep"

    def __init__(self, result):
        self._result = result

    # LINE ====================================================================
    def _resolve_axis(self, ax):

        if ax is None:
            coords_number = len(self._result.pcoords_)

            fig, ax = plt.subplots(1, coords_number, sharey=True)

            size_x, size_y = fig.get_size_inches()
            fig.set_size_inches(size_x * coords_number, size_y)

        if not isinstance(ax, Iterable):
            ax = np.array([ax])

        return ax

    def _resolve_fix_dimension(self, dim, dim_value, xa):
        if dim_value in HEURISTICS:
            dim_value = HEURISTICS[dim_value]
        if callable(dim_value):
            dim_value = dim_value(dim, xa)
        return dim_value

    def _complete_dimension(self, xa, dim, n, scalar_dim=True):
        # the original array must be in the final data
        completed = [xa]

        zxa = xa.copy()
        zxa.values = np.zeros_like(zxa.values)

        for idx in np.arange(-n, n + 1):
            if idx == 0:
                continue

            nxa = zxa.copy()
            nxa[dim] = idx if scalar_dim else [idx]
            completed.append(nxa)

        return xr.combine_nested(completed, dim)

    def linep(self, time="mean_max_value", **kwargs):

        xa = self._result.to_xarray()

        time = self._resolve_fix_dimension("times", time, xa)
        axes = self._resolve_axis(kwargs.pop("ax", None))
        has_single_position = len(self._result.positions_) == 1

        if has_single_position:
            xa = self._complete_dimension(xa, "positions", 25, scalar_dim=True)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes):

            df = xa.sel(positions_coordinates=coord, times=time).to_dataframe()

            sns.lineplot(
                x="positions",
                y="values",
                hue="modes",
                data=df,
                ax=ax,
                **kwargs,
            )

        model_name = self._result.mname
        ax.set_title(f"{model_name} - Time {time}")
        ax.legend()

        return ax

    def linet(self, position="mean_max_value", **kwargs):

        xa = self._result.to_xarray()
        position = self._resolve_fix_dimension("positions", position, xa)
        axes = self._resolve_axis(kwargs.pop("ax", None))
        has_single_time = len(self._result.times_) == 1

        if has_single_time:
            xa = self._complete_dimension(xa, "times", 25, scalar_dim=False)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes):

            df = xa.sel(
                positions_coordinates=coord, positions=position
            ).to_dataframe()

            sns.lineplot(
                x="times",
                y="values",
                hue="modes",
                data=df,
                ax=ax,
                **kwargs,
            )

        model_name = self._result.mname
        ax.set_title(f"{model_name} - Position {position}")
        ax.legend()

        return ax
