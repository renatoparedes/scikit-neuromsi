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
from skneuromsi.core.constants import D_MODES, D_POSITIONS, D_TIMES, XA_NAME

import xarray as xr

from ..utils import AccessorABC


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


class ResultPlotter(AccessorABC):
    """Make plots of Result.

    Kind of plot to produce:



    """

    _default_kind = "line_positions"

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

    # API======================================================================

    def line_positions(self, time=None, **kwargs):

        if time is None:
            time = self._result.stats.dimmax()[D_TIMES]

        axes = self._resolve_axis(kwargs.pop("ax", None))
        has_single_position = len(self._result.positions_) == 1
        position_res = self._result.position_res

        xa = self._result.to_xarray()

        if has_single_position:
            xa = self._complete_dimension(xa, D_POSITIONS, 25, scalar_dim=True)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes):

            df = xa.sel(positions_coordinates=coord, times=time).to_dataframe()

            sns.lineplot(
                x=D_POSITIONS,
                y=XA_NAME,
                hue=D_MODES,
                data=df,
                ax=ax,
                **kwargs,
            )

            # rescale the ticks by resolution
            ticks = ax.get_xticks()
            labels = [float(t) * position_res for t in ticks]
            ax.set_xticks(ticks)  # without this a warning will be raised
            ax.set_xticklabels(labels)

        model_name = self._result.mname
        ax.set_title(f"{model_name} - Time {time}")
        ax.legend()

        return ax

    linep = line_positions

    def line_times(self, position=None, **kwargs):

        if position is None:
            position = self._result.stats.dimmax()[D_POSITIONS]

        axes = self._resolve_axis(kwargs.pop("ax", None))
        has_single_time = len(self._result.times_) == 1
        time_res = self._result.time_res

        xa = self._result.to_xarray()

        if has_single_time:
            xa = self._complete_dimension(xa, D_TIMES, 25, scalar_dim=False)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes):

            df = xa.sel(
                positions_coordinates=coord, positions=position
            ).to_dataframe()

            sns.lineplot(
                x=D_TIMES,
                y=XA_NAME,
                hue=D_MODES,
                data=df,
                ax=ax,
                **kwargs,
            )

            # rescale the ticks by resolution
            ticks = ax.get_xticks()
            labels = [float(t) * time_res for t in ticks]
            ax.set_xticks(ticks)  # without this a warning will be raised
            ax.set_xticklabels(labels)

        model_name = self._result.mname
        ax.set_title(f"{model_name} - Position {position}")
        ax.legend()

        return ax

    linet = line_times
