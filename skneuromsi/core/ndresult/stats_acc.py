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

"""Stats helper for the Result object."""


# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

from ..constants import DIMENSIONS
from ...utils import AccessorABC

# =============================================================================
# STATS ACCESSOR
# =============================================================================


class ResultStatsAccessor(AccessorABC):
    """Calculate basic statistics of the result.

    Kind of statistic to produce:
    - describe
    - count
    - mean
    - std
    - min
    - max
    - dimmin
    - dimmax
    - quantile

    Parameters
    ----------
    result : NDResult
        The NDResult object for which to calculate statistics.

    """

    _default_kind = "describe"

    def __init__(self, result):
        self._result = result

    def _to_xarray(self, modes, times, positions, coordinates):
        """Convert the NDResult to an xarray.DataArray with specified filters.

        Parameters
        ----------
        modes : array-like or None
            The modes to include in the filtered xarray.DataArray.
        times : array-like or None
            The times to include in the filtered xarray.DataArray.
        positions : array-like or None
            The positions to include in the filtered xarray.DataArray.
        coordinates : array-like or None
            The coordinates to include in the filtered xarray.DataArray.

        Returns
        -------
        xarray.DataArray
            The filtered xarray.DataArray.

        """
        xa = self._result.to_xarray()
        flt = {
            dim_name: dim_flt
            for dim_name, dim_flt in zip(
                DIMENSIONS, (modes, times, positions, coordinates)
            )
            if dim_flt is not None
        }
        return xa.sel(flt) if flt else xa

    def count(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        """Count the number of elements in the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the count.
        times : array-like or None, optional
            The times to include in the count.
        positions : array-like or None, optional
            The positions to include in the count.
        coordinates : array-like or None, optional
            The coordinates to include in the count.

        Returns
        -------
        int
            The number of elements in the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return int(xa.count().to_numpy())

    def mean(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        """Calculate the mean value of the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the mean calculation.
        times : array-like or None, optional
            The times to include in the mean calculation.
        positions : array-like or None, optional
            The positions to include in the mean calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the mean calculation.

        Returns
        -------
        float
            The mean value of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.mean().to_numpy())

    def std(self, *, modes=None, times=None, positions=None, coordinates=None):
        """Calculate the standard deviation of the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the standard deviation calculation.
        times : array-like or None, optional
            The times to include in the standard deviation calculation.
        positions : array-like or None, optional
            The positions to include in the standard deviation calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the standard deviation calculation.

        Returns
        -------
        float
            The standard deviation of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.std().to_numpy())

    def min(  # noqa
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        """Calculate the minimum value of the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the minimum value calculation.
        times : array-like or None, optional
            The times to include in the minimum value calculation.
        positions : array-like or None, optional
            The positions to include in the minimum value calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the minimum value calculation.

        Returns
        -------
        float
            The minimum value of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.min().to_numpy())

    def dimmin(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        """Calculate the minimum value and corresponding dimensions of the \
        NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the minimum value calculation.
        times : array-like or None, optional
            The times to include in the minimum value calculation.
        positions : array-like or None, optional
            The positions to include in the minimum value calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the minimum value calculation.

        Returns
        -------
        pandas.Series
            A series containing the minimum value and corresponding dimensions
            of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)

        data = {}

        for dim, min_idx_xa in xa.argmin(...).items():
            arr = xa[dim].to_numpy()
            min_idx = min_idx_xa.to_numpy()
            data[dim] = arr[min_idx]

        data["values"] = xa.min(...).to_numpy()

        return pd.Series(data, name="min")

    def max(  # noqa
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):  # noqa
        """Calculate the maximum value of the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the maximum value calculation.
        times : array-like or None, optional
            The times to include in the maximum value calculation.
        positions : array-like or None, optional
            The positions to include in the maximum value calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the maximum value calculation.

        Returns
        -------
        float
            The maximum value of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.max().to_numpy())

    def dimmax(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        """Calculate the maximum value and corresponding dimensions of \
        the NDResult.

        Parameters
        ----------
        modes : array-like or None, optional
            The modes to include in the maximum value calculation.
        times : array-like or None, optional
            The times to include in the maximum value calculation.
        positions : array-like or None, optional
            The positions to include in the maximum value calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the maximum value calculation.

        Returns
        -------
        pandas.Series
            A series containing the maximum value and corresponding dimensions
            of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)

        data = {}

        for dim, max_idx_xa in xa.argmax(...).items():
            arr = xa[dim].to_numpy()
            max_idx = max_idx_xa.to_numpy()
            data[dim] = arr[max_idx]

        data["values"] = xa.max(...).to_numpy()

        return pd.Series(data, name="max")

    def quantile(
        self,
        q=0.25,
        *,
        modes=None,
        times=None,
        positions=None,
        coordinates=None,
        **kwargs,
    ):
        """Calculate the quantile value(s) of the NDResult.

        Parameters
        ----------
        q : float or array-like, optional
            The quantile(s) to calculate. Default is 0.25.
        modes : array-like or None, optional
            The modes to include in the quantile calculation.
        times : array-like or None, optional
            The times to include in the quantile calculation.
        positions : array-like or None, optional
            The positions to include in the quantile calculation.
        coordinates : array-like or None, optional
            The coordinates to include in the quantile calculation.
        **kwargs
            Additional keyword arguments to pass to
            xarray.DataArray.quantile().

        Returns
        -------
        numpy.ndarray
            The calculated quantile value(s) of the filtered NDResult.

        """
        xa = self._to_xarray(modes, times, positions, coordinates)
        return xa.quantile(q=q, **kwargs).to_numpy()

    def describe(self, percentiles=None):
        """Generate descriptive statistics of the NDResult.

        Parameters
        ----------
        percentiles : array-like or None, optional
            The percentiles to include in the descriptive statistics.
            Default is [0.25, 0.5, 0.75].

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing descriptive statistics of the NDResult.

        """
        percentiles = [0.25, 0.5, 0.75] if percentiles is None else percentiles

        xa = self._result.to_xarray()
        data = {}

        data["count"] = xa.count().to_numpy()
        data["mean"] = xa.mean().to_numpy()
        data["std"] = xa.std().to_numpy()
        data["min"] = xa.min().to_numpy()

        for perc, pvalue in zip(
            percentiles, xa.quantile(percentiles).to_numpy()
        ):
            pkey = f"{int(perc * 100)}%"
            data[pkey] = pvalue

        data["max"] = xa.max().to_numpy()

        return pd.Series(data, name="describe", dtype=float).to_frame()
