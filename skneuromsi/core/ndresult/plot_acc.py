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

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import xarray as xr

from ..constants import D_MODES, D_POSITIONS, D_TIMES, XA_NAME
from ...utils import AccessorABC


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


class ResultPlotter(AccessorABC):
    """Make plots of Result.

    Kind of plot to produce:
    - line_positions
    - line_times

    Parameters
    ----------
    result : NDResult
        The NDResult object for which to create plots.

    """

    _default_kind = "line_positions"

    def __init__(self, result):
        self._result = result

    # LINE ====================================================================
    def _resolve_axis(self, ax):
        """Resolve the axis for plotting.

        If `ax` is None, create a new figure and axis based on the number of
        position coordinates in the result. Otherwise, ensure `ax` is an
        iterable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            The axis to plot on. If None, a new figure and axis will
            be created.

        Returns
        -------
        ax : numpy.ndarray
            The resolved axis as a numpy array.

        """
        if ax is None:
            coords_number = len(self._result.pcoords_)

            fig, ax = plt.subplots(1, coords_number, sharey=True)

            size_x, size_y = fig.get_size_inches()
            fig.set_size_inches(size_x * coords_number, size_y)

        if not isinstance(ax, Iterable):
            ax = [ax]

        return np.asarray(ax)

    def _complete_dimension(self, xa, dim, n, scalar_dim=True):
        """Complete a dimension in the xarray.DataArray.

        This method creates a new xarray.DataArray with the specified dimension
        `dim` completed by adding zero-valued entries around the original data.

        Parameters
        ----------
        xa : xarray.DataArray
            The input xarray.DataArray.
        dim : str
            The dimension to complete.
        n : int
            The number of entries to add on each side of the original data.
        scalar_dim : bool, optional
            Whether the dimension is scalar (True) or not (False).
            Default is True.

        Returns
        -------
        xarray.DataArray
            The completed xarray.DataArray.

        """
        import ipdb; ipdb.set_trace()
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

    def _scale_xtickslabels(self, *, limits, ticks, single_value):
        """Scale the x-tick labels based on the provided limits and ticks.

        Parameters
        ----------
        limits : tuple
            The lower and upper limits for scaling the x-tick labels.
        ticks : array-like
            The original tick positions.
        single_value : bool
            Whether there is a single value in the data.

        Returns
        -------
        labels : numpy.ndarray
            The scaled x-tick labels.

        """
        ll, tl = np.sort(limits)
        ticks_array = np.asarray(ticks, dtype=float)

        tmin, tmax = np.min(ticks_array), np.max(ticks_array)
        new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
        labels = np.array([f"{t:.2f}" for t in new_ticks])

        if single_value:
            mask = np.ones_like(labels, dtype=bool)
            mask[len(mask) // 2] = False
            labels[mask] = ""

        return labels

    # API======================================================================

    def line_positions(self, time=None, **kwargs):
        """Create a line plot of positions at a specific time.

        Parameters
        ----------
        time : float or None, optional
            The time at which to plot the positions. If None, the maximum time
            from the result will be used. Default is None.
        **kwargs
            Additional keyword arguments to pass to seaborn.lineplot().

        Returns
        -------
        axes : numpy.ndarray
            The plotted axes.

        """
        if time is None:
            time = self._result.stats.dimmax()[D_TIMES]

        axes = self._resolve_axis(kwargs.pop("ax", None)).flatten()
        has_single_position = len(self._result.positions_) == 1
        position_range = self._result.position_range

        xa = self._result.to_xarray()

        if has_single_position:
            xa = self._complete_dimension(xa, D_POSITIONS, 25, scalar_dim=True)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes, strict=True):

            df = xa.sel(positions_coordinates=coord, times=time).to_dataframe()

            sns.lineplot(
                x=D_POSITIONS,
                y=XA_NAME,
                hue=D_MODES,
                data=df,
                ax=ax,
                legend=(ax == axes[-1]),  # the last ax has the legend
                **kwargs,
            )

            # rescale the ticks by resolution
            ticks = ax.get_xticks()
            labels = self._scale_xtickslabels(
                limits=position_range,
                ticks=ticks,
                single_value=has_single_position,
            )
            ax.set_xticks(ticks)  # without this a warning will be raised
            ax.set_xticklabels(labels)

            # title
            ax.set_title(coord)

        # retrieve the figure
        figure = ax.get_figure()
        figure.suptitle(f"{self._result.mname} - Time {time}")

        return axes

    linep = line_positions

    def line_times(self, position=None, **kwargs):
        """Create a line plot of time series at a specific position.

        Parameters
        ----------
        position : float or None, optional
            The position at which to plot the time series. If None, the
            position with the maximum value from the result will be used.
            Default is None.
        **kwargs
            Additional keyword arguments to pass to seaborn.lineplot().

        Returns
        -------
        ax : numpy.ndarray
            The plotted axes.

        """
        if position is None:
            position = self._result.stats.dimmax()[D_POSITIONS]

        axes = self._resolve_axis(kwargs.pop("ax", None))
        has_single_time = len(self._result.times_) == 1
        time_range = self._result.time_range

        xa = self._result.to_xarray()

        if has_single_time:
            xa = self._complete_dimension(xa, D_TIMES, 25, scalar_dim=False)

        kwargs.setdefault("alpha", 0.75)

        for coord, ax in zip(self._result.pcoords_, axes, strict=True):
            df = xa.sel(
                positions_coordinates=coord, positions=position
            ).to_dataframe()

            sns.lineplot(
                x=D_TIMES,
                y=XA_NAME,
                hue=D_MODES,
                data=df,
                ax=ax,
                legend=(ax == axes[-1]),  # the last ax has the legend
                **kwargs,
            )

            # rescale the ticks by resolution
            ticks = ax.get_xticks()
            labels = self._scale_xtickslabels(
                limits=time_range, ticks=ticks, single_value=has_single_time
            )
            ax.set_xticks(ticks)  # without this a warning will be raised
            ax.set_xticklabels(labels)

            # title
            ax.set_title(coord)

        figure = ax.get_figure()
        figure.suptitle(f"{self._result.mname} - Position {position}")

        return axes

    linet = line_times
