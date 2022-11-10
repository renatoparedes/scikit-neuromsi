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

"""Stats helper for the Result object."""


# =============================================================================
# IMPORTS
# =============================================================================k

import pandas as pd

from .constants import DIMENSIONS
from ..utils import AccessorABC


# =============================================================================
# STATS ACCESSOR
# =============================================================================


class ResultStatsAccessor(AccessorABC):
    """Calculate basic statistics of the  result

    Kind of statistic to produce:


    """

    _default_kind = "describe"

    def __init__(self, result):
        self._result = result

    def _to_xarray(self, modes, times, positions, coordinates):
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
        xa = self._to_xarray(modes, times, positions, coordinates)
        return int(xa.count().to_numpy())

    def mean(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.mean().to_numpy())

    def std(self, *, modes=None, times=None, positions=None, coordinates=None):
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.std().to_numpy())

    def min(self, *, modes=None, times=None, positions=None, coordinates=None):  # noqa
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.min().to_numpy())

    def dimmin(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
        xa = self._to_xarray(modes, times, positions, coordinates)

        data = {}

        for dim, min_idx_xa in xa.argmin(...).items():
            arr = xa[dim].to_numpy()
            min_idx = min_idx_xa.to_numpy()
            data[dim] = arr[min_idx]

        data["values"] = xa.min(...).to_numpy()

        return pd.Series(data, name="min")

    def max(self, *, modes=None, times=None, positions=None, coordinates=None):  # noqa
        xa = self._to_xarray(modes, times, positions, coordinates)
        return float(xa.max().to_numpy())

    def dimmax(
        self, *, modes=None, times=None, positions=None, coordinates=None
    ):
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
        xa = self._to_xarray(modes, times, positions, coordinates)
        return xa.quantile(q=q, **kwargs).to_numpy()

    def describe(self, percentiles=None):
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
