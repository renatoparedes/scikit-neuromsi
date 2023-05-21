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
from collections.abc import Sequence

import methodtools

import numpy as np

import pandas as pd

from tqdm.auto import tqdm

import seaborn as sns

from ..utils import AccessorABC, Bunch

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


def _describe_modes(ndres):
    modes = ndres.get_modes()
    describe = modes.describe()
    describe.loc["var"] = modes.var()
    describe.loc["kurtosis"] = modes.kurtosis()
    describe.loc["skew"] = modes.skew()
    return describe


def _make_metadata_cache(ndresults, progress_cls):
    mnames = []
    mtypes = []
    nmaps = []
    time_ranges = []
    position_ranges = []
    time_resolutions = []
    position_resolutions = []
    run_parameters = []
    extras = []
    causes = []
    modes_describes = []

    if progress_cls:
        ndresults = progress_cls(
            iterable=ndresults, desc="Collecting metadata"
        )

    for ndres in ndresults:
        mnames.append(ndres.mname)
        mtypes.append(ndres.mtype)
        nmaps.append(ndres.nmap_)
        time_ranges.append(ndres.time_range)
        position_ranges.append(ndres.position_range)
        time_resolutions.append(ndres.time_res)
        position_resolutions.append(ndres.position_res)
        run_parameters.append(ndres.run_params.to_dict())
        extras.append(ndres.extra_.to_dict())
        causes.append(ndres.causes_)
        modes_describes.append(_describe_modes(ndres))

    # all the modes are the same fo take the last one
    modes = ndres.modes_

    cache = Bunch(
        "ndcollectrion_metadata_cache",
        {
            "modes": modes,
            "mnames": np.asarray(mnames),
            "mtypes": np.asarray(mtypes),
            "nmaps": np.asarray(nmaps),
            "time_ranges": np.asarray(time_ranges),
            "position_ranges": np.asarray(position_ranges),
            "time_resolutions": np.asarray(time_resolutions),
            "position_resolutions": np.asarray(position_resolutions),
            "run_parameters": np.asarray(run_parameters),
            "extras": np.asarray(extras),
            "causes": np.asarray(causes),
            "modes_describes": tuple(modes_describes),
        },
    )

    return cache


class NDResultCollection(Sequence):
    def __init__(self, name, results, *, tqdm_cls=tqdm):
        self._len = len(results)
        if not self._len:
            cls_name = type(self).__name__
            raise ValueError(f"Empty {cls_name} not allowed")

        self._name = str(name)
        self._ndresults = results
        self._progress_cls = tqdm_cls
        self._metadata_cache = _make_metadata_cache(
            results, progress_cls=tqdm_cls
        )

    # Because is a Sequence ==================================================
    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._len

    def __getitem__(self, idxs):
        return self._ndresults.__getitem__(idxs)

    @methodtools.lru_cache(maxsize=None)
    @property
    def plot(self):
        return NDResultCollectionPlotter(self)

    @property
    def modes_(self):
        return self._metadata_cache.modes

    def __repr__(self):
        cls_name = type(self).__name__
        name = self.name
        length = len(self)
        storage = type(self._ndresults).__name__
        return f"<{cls_name} {name!r} len={length} storage={storage}>"

    # CAUSES ==================================================================

    def disparity_matrix(self):
        df = pd.DataFrame(list(self._metadata_cache.run_parameters))
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

    def causes_by_attribute(self, *, attribute=None):
        attribute = self._get_attribute_by(attribute)
        run_params = self._metadata_cache.run_parameters
        causes = self._metadata_cache.causes

        columns = defaultdict(list)
        for rp, causes in zip(run_params, causes):
            columns[("Attributes", attribute)].append(rp[attribute])
            columns[("", "Causes")].append(causes)

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

    # BIAS ====================================================================
    def modes_variance(self):
        variances_by_res = []
        for modes_describe in self._metadata_cache.modes_describes:
            variances_by_res.append(modes_describe.loc["var"])
        variances = pd.DataFrame.from_records(variances_by_res).sum()
        variances.name = "var"
        return variances



    def bias(self, *, mode=None):
        mode = self._coherce_mode(mode)
