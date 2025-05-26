#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
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

import pandas as pd

import seaborn as sns

from ..constants import (
    D_MODES,
    D_POSITIONS,
    D_POSITIONS_COORDINATES,
    D_TIMES,
    XA_NAME,
)
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

    def _complete_index_level(self, *, df, level, center, fill_value):
        """Complete a DataFrame index level with missing values around a \
        center point.

        This function fills in missing values in a specified index level
        of a DataFrame, creating a symmetric range of values around a center
        point.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be completed.
        level : str
            The name of the index level to be completed.
        center : int
            The center value around which to complete the index.
        fill_value : scalar
            The value to use for filling new rows.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with the completed index level.

        Notes
        -----
        The function creates a range of 51 values centered around the given
        center value, excluding the center itself. It then uses these values
        to complete the specified index level.

        """
        # Create a range of values to complete the index,
        # centered around the given center value
        values_to_complete = np.arange(center - 25, center + 26)
        values_to_complete = values_to_complete[values_to_complete != center]

        # Prepare new index values for all levels
        index_names, new_index_values = [], []
        for lname, lvalues in zip(df.index.names, df.index.levels):
            lvalues = lvalues.to_numpy()

            # Replace the values for the specified level with the new range
            if level == lname:
                lvalues = values_to_complete

            index_names.append(lname)
            new_index_values.append(lvalues)

        # Create a new DataFrame with the expanded index and fill it
        # with the specified fill_value
        for_complete_df = pd.DataFrame(
            fill_value,
            columns=df.columns.copy(),
            index=pd.MultiIndex.from_product(
                new_index_values, names=index_names
            ),
        )

        # Concatenate the original DataFrame with the new one to
        # create the completed DataFrame
        completed_df = pd.concat((df, for_complete_df))
        return completed_df

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

        kwargs.setdefault("alpha", 0.75)

        df = xa.sel(times=time).to_dataframe()
        if has_single_position:
            df = self._complete_index_level(
                df=df,
                level=D_POSITIONS,
                center=self._result.positions_[0],
                fill_value={"times": time, "values": 0},
            )

        for pcoord, ax in zip(self._result.pcoords_, axes, strict=True):
            pcoord_df = df.xs(pcoord, level=D_POSITIONS_COORDINATES)

            sns.lineplot(
                x=D_POSITIONS,
                y=XA_NAME,
                hue=D_MODES,
                data=pcoord_df,
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
            ax.set_title(pcoord)

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

        kwargs.setdefault("alpha", 0.75)

        df = xa.sel(positions=position).to_dataframe()
        if has_single_time:
            df = self._complete_index_level(
                df=df,
                level=D_TIMES,
                center=self._result.times_[0],
                fill_value={"positions": position, "values": 0},
            )

        for pcoord, ax in zip(self._result.pcoords_, axes, strict=True):
            pcoord_df = df.xs(pcoord, level=D_POSITIONS_COORDINATES)

            sns.lineplot(
                x=D_TIMES,
                y=XA_NAME,
                hue=D_MODES,
                data=pcoord_df,
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
            ax.set_title(pcoord)

        figure = ax.get_figure()
        figure.suptitle(f"{self._result.mname} - Position {position}")

        return axes

    linet = line_times
