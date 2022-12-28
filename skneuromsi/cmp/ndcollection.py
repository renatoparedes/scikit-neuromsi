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

import functools
from collections import defaultdict

import numpy as np

import pandas as pd

import seaborn as sns

from ..core import NDResult
from ..utils import AccessorABC

# =============================================================================
# PLOTTER
# =============================================================================


class NDResultCollectionPlotter(AccessorABC):

    _default_kind = "unity_report"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def _line_report(self, report, ax, kws):
        attribute = report.columns[0]
        x = report.index
        y = report[attribute]

        kws.setdefault("label", attribute)
        ax = sns.lineplot(x=x, y=y, ax=ax, **kws)

        return ax

    def n_report(self, n, *, attribute=None, ax=None, **kws):
        the_report = self._nd_collection.n_report(n, attribute=attribute)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel(f"Proportion of {n} causes")
        return ax

    def unity_report(self, *, attribute=None, ax=None, **kws):
        the_report = self._nd_collection.unity_report(attribute=attribute)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Proportion of unit causes")
        return ax

    def mean_report(self, *, attribute=None, ax=None, **kws):
        the_report = self._nd_collection.mean_report(attribute=attribute)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Mean of causes")
        return ax


# =============================================================================
# RESULT COLLECTION
# =============================================================================


class NDResultCollection:
    def __init__(self, ndresults, name=None):

        self._name = str(name)
        self._ndresults = tuple(ndresults)

        if not all(isinstance(e, NDResult) for e in self._ndresults):
            raise TypeError(
                "All elements of 'ndresults' must be instances of NDResult"
            )

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        return NDResultCollectionPlotter(self)

    def __repr__(self):
        cls_name = type(self).__name__
        name = self._name
        length = len(self._ndresults)
        return f"<{cls_name} name={name!r}, len={length}>"

    def __getitem__(self, slice):
        return self._ndresults.__getitem__(slice)

    def __len__(self):
        return len(self._ndresults)

    def disparity_matrix(self):
        df = pd.DataFrame(r.rp for r in self._ndresults)
        df.index.name = "Iteration"
        df.columns.name = "Attributes"
        return df

    def changing_attributes(self):
        dm = self.disparity_matrix()
        uniques = dm.apply(np.unique)
        changes = uniques.apply(len) != 1
        changes.name = "Changes"
        return changes

    def _get_attribute_by(self, attribute):
        if attribute is None:
            wpc = self.changing_attributes()
            candidates = wpc[wpc].index.to_numpy()
            candidates_len = len(candidates)
            if candidates_len != 1:
                raise ValueError(
                    "The value of attribute is ambiguous since it has "
                    f"{candidates_len} candidates: {candidates}"
                )
            attribute = candidates[0]
        return attribute

    # CAUSES ==================================================================
    def causes_by_attribute(self, *, attribute=None):
        attribute = self._get_attribute_by(attribute)

        columns = defaultdict(list)
        for ndres in self._ndresults:

            columns[("Attributes", attribute)].append(ndres.rp[attribute])
            columns[("", "Causes")].append(ndres.causes_)

        cdf = pd.DataFrame.from_dict(columns)

        cdf.index.name = "Iteration"
        cdf["Attributes"] -= cdf["Attributes"].min()

        # put al the attributes together
        cdf = cdf[np.sort(cdf.columns)[::-1]]

        return cdf

    def unique_causes(self, *, attribute=None):
        cba = self.causes_by_attribute(attribute=attribute)
        return cba[("", "Causes")].unique()

    def n_report(self, n, *, attribute=None):
        attribute = self._get_attribute_by(attribute)
        cdf = self.causes_by_attribute(attribute=attribute)

        values = cdf[("Attributes", attribute)]
        crosstab = pd.crosstab(values, cdf["", "Causes"])
        n_ity = crosstab.get(n, 0) / crosstab.sum(axis="columns")

        the_report = pd.DataFrame(n_ity, columns=[attribute])
        the_report.index.name = "Disparity"
        the_report.columns.name = "Causes"

        return the_report

    def unity_report(self, *, attribute=None):
        return self.n_report(1, attribute=attribute)

    def mean_report(self, *, attribute=None):
        attribute = self._get_attribute_by(attribute)
        cdf = self.causes_by_attribute(attribute=attribute)

        groups = cdf.groupby(("Attributes", attribute))
        report = groups.mean()

        report.index.name = "Disparity"

        report.columns = [attribute]
        report.columns.name = "Causes"

        return report

    def describe_causes(self, *, attribute=None):
        attribute = self._get_attribute_by(attribute)
        cdf = self.causes_by_attribute(attribute=attribute)

        groups = cdf.groupby(("Attributes", attribute))
        report = groups.describe()

        report.index.name = "Disparity"

        columns = report.columns
        report.columns = pd.MultiIndex.from_product(
            [[attribute], columns.levels[-1]], names=["Causes", None]
        )
        report.columns.name = "Causes"

        return report
