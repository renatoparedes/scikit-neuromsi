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

from ..utils import AccessorABC

# =============================================================================
# STATS ACCESSOR
# =============================================================================


class ResultStatsAccessor(AccessorABC):
    """Calculate basic statistics of the  result

    Kind of statistic to produce:


    """

    # # The list of methods that can be accessed of the subjacent dataframe.
    # _DF_WHITELIST = (
    #     "corr",
    #     "cov",
    #     "describe",
    #     "kurtosis",
    #     "mad",
    #     "max",
    #     "mean",
    #     "median",
    #     "min",
    #     "pct_change",
    #     "quantile",
    #     "sem",
    #     "skew",
    #     "std",
    #     "var",
    # )

    _default_kind = "describe"

    def __init__(self, result):
        self._result = result

    def count(self):
        xa = self._result.to_xarray()
        return int(xa.count().to_numpy())

    def mean(self):
        xa = self._result.to_xarray()
        return float(xa.mean().to_numpy())

    def std(self):
        xa = self._result.to_xarray()
        return float(xa.std().to_numpy())

    def min(self):
        xa = self._result.to_xarray()
        return float(xa.min().to_numpy())

    def dimmin(self):
        xa = self._result.to_xarray()

        data = {}

        for dim, min_idx_xa in xa.argmin(...).items():
            arr = xa[dim].to_numpy()
            min_idx = min_idx_xa.to_numpy()
            data[dim] = arr[min_idx]

        data["values"] = xa.min(...).to_numpy()

        return pd.Series(data, name="min")

    def max(self):
        xa = self._result.to_xarray()
        return float(xa.max().to_numpy())

    def dimmax(self):
        xa = self._result.to_xarray()

        data = {}

        for dim, max_idx_xa in xa.argmax(...).items():
            arr = xa[dim].to_numpy()
            max_idx = max_idx_xa.to_numpy()
            data[dim] = arr[max_idx]

        data["values"] = xa.max(...).to_numpy()

        return pd.Series(data, name="max")

    def quantile(self, q=0.25, **kwargs):
        xa = self._result.to_xarray()
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

        return pd.Series(data, name="describe")
