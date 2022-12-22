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

# =============================================================================
# PLOTTER
# =============================================================================


class NDResultCollectionPlotter:
    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def unity_report(self, ax=None, **kws):
        the_report = self._nd_collection.unity_report()
        ax = sns.lineplot(data=the_report, ax=ax, **kws)
        ax.set_title("Unity Report")
        ax.set_ylabel("causes=1 (Proportion)")
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

    def disparity_matrix(self):
        df = pd.DataFrame(r.rp for r in self._ndresults)
        df.index.name = "Iteration"
        df.columns.name = "Attributes"
        return df

    def which_attrs_change(self):
        dm = self.disparity_matrix()
        uniques = dm.apply(np.unique)
        changes = uniques.apply(len) != 1
        changes.name = "Changes"
        return changes

    def changing_attrs(self):
        wpc = self.which_attrs_change()
        return wpc[wpc].index.to_numpy()

    def causes_by_attr(self, attributes=None):
        attributes = (
            self.changing_attrs() if attributes is None else attributes
        )

        columns = defaultdict(list)
        for ndres in self._ndresults:
            for attr in attributes:
                columns[("Attributes", attr)].append(ndres.rp[attr])
            columns[("", "Causes")].append(ndres.causes_)

        cdf = pd.DataFrame.from_dict(columns)

        cdf.index.name = "Iteration"
        cdf["Attributes"] -= cdf["Attributes"].min()

        # put al the attributes together
        cdf = cdf[np.sort(cdf.columns)[::-1]]

        return cdf

    def unity_report(self, attributes=None):
        attributes = (
            self.changing_attrs() if attributes is None else attributes
        )
        cdf = self.causes_by_attr(attributes)
        columns = {}
        for attr in attributes:
            values = cdf[("Attributes", attr)]
            attr_ctab = pd.crosstab(values, cdf["", "Causes"])
            attr_unity = attr_ctab[1] / attr_ctab.sum(axis="columns")
            columns[attr] = attr_unity

        the_report = pd.DataFrame.from_dict(columns)
        the_report.index.name = "Disparity"
        the_report.columns.name = "Attributes"

        return the_report

    def mean_report(self, attributes=None):
        attributes = (
            self.changing_attrs() if attributes is None else attributes
        )
        cdf = self.causes_by_attr(attributes)
        columns = {}
        for attr in attributes:
            pass
