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

""""""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import defaultdict

import numpy as np

import pandas as pd

from ...utils import AccessorABC


# =============================================================================
# NDResultCauses ACC
# =============================================================================


class NDResultCausesAcc(AccessorABC):
    _default_kind = "unity_report"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def causes_by_parameter(self, *, parameter=None):
        nd_collection = self._nd_collection

        parameter = nd_collection.coerce_parameter(parameter)

        cache = nd_collection._metadata_cache
        run_params_values = cache.run_parameters_values
        causes = cache.causes

        columns = defaultdict(list)
        for rp_value, causes in zip(run_params_values, causes):
            columns[("Parameters", parameter)].append(rp_value[parameter])
            columns[("", "Causes")].append(causes)

        cdf = pd.DataFrame.from_dict(columns)

        cdf.index.name = "Iteration"
        cdf["Parameters"] -= cdf["Parameters"].min()

        # put al the parameters together
        cdf = cdf[np.sort(cdf.columns)[::-1]]

        return cdf

    def unique_causes(self, *, parameter=None):
        cba = self.causes_by_parameter(parameter=parameter)
        return cba[("", "Causes")].unique()

    def n_report(self, n, *, parameter=None):
        nd_collection = self._nd_collection

        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        values = cdf[("Parameters", parameter)]
        crosstab = pd.crosstab(values, cdf["", "Causes"])
        n_ity = crosstab.get(n, 0) / crosstab.sum(axis="columns")

        the_report = pd.DataFrame(n_ity, columns=[parameter])
        the_report.index.name = "Disparity"
        the_report.columns.name = "Causes"

        return the_report

    def unity_report(self, *, parameter=None):
        return self.n_report(1, parameter=parameter)

    def mean_report(self, *, parameter=None):
        nd_collection = self._nd_collection
        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        groups = cdf.groupby(("Parameters", parameter))
        report = groups.mean()

        report.index.name = "Disparity"

        report.columns = [parameter]
        report.columns.name = "Causes"

        return report

    def describe_causes(self, *, parameter=None):
        nd_collection = self._nd_collection
        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        groups = cdf.groupby(("Parameters", parameter))
        report = groups.describe()

        report.index.name = "Disparity"

        columns = report.columns
        report.columns = pd.MultiIndex.from_product(
            [[parameter], columns.levels[-1]], names=["Causes", None]
        )
        report.columns.name = "Causes"

        return report
