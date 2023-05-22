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
        parameter = report.columns[0]
        x = report.index
        y = report[parameter]

        kws.setdefault("label", parameter)
        ax = sns.lineplot(x=x, y=y, ax=ax, **kws)

        return ax

    def n_report(self, n, *, parameter=None, ax=None, **kws):
        the_report = self._nd_collection.n_report(n, parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel(f"Proportion of {n} causes")
        return ax

    def unity_report(self, *, parameter=None, ax=None, **kws):
        the_report = self._nd_collection.unity_report(parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Proportion of unit causes")
        return ax

    def mean_report(self, *, parameter=None, ax=None, **kws):
        the_report = self._nd_collection.mean_report(parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Mean of causes")
        return ax


# =============================================================================
# RESULT COLLECTION
# =============================================================================


def _modes_describe(ndres):
    modes = ndres.get_modes()
    return {"var": modes.var()}


def _make_metadata_cache(ndresults, progress_cls):
    mnames = []
    mtypes = []
    nmaps = []
    time_ranges = []
    position_ranges = []
    time_resolutions = []
    position_resolutions = []
    run_parameters_values = []
    extras = []
    causes = []
    modes_variances = []

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
        run_parameters_values.append(ndres.run_params.to_dict())
        extras.append(ndres.extra_.to_dict())
        causes.append(ndres.causes_)

        modes_describe_dict = _modes_describe(ndres)
        modes_variances.append(modes_describe_dict["var"])

    # all the run_parameters/modes are the same fo take the last one
    modes = ndres.modes_
    run_parameters = tuple(ndres.run_params.to_dict())

    cache = Bunch(
        "ndcollection_metadata_cache",
        {
            "modes": modes,
            "run_parameters": run_parameters,
            "mnames": np.asarray(mnames),
            "mtypes": np.asarray(mtypes),
            "nmaps": np.asarray(nmaps),
            "time_ranges": np.asarray(time_ranges),
            "position_ranges": np.asarray(position_ranges),
            "time_resolutions": np.asarray(time_resolutions),
            "position_resolutions": np.asarray(position_resolutions),
            "run_parameters_values": np.asarray(run_parameters_values),
            "extras": np.asarray(extras),
            "causes": np.asarray(causes),
            "modes_variances_sum": sum(modes_variances),
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

    @property
    def run_parameters_(self):
        return self._metadata_cache.run_parameters

    def __repr__(self):
        cls_name = type(self).__name__
        name = self.name
        length = len(self)
        storage = type(self._ndresults).__name__
        return f"<{cls_name} {name!r} len={length} storage={storage!r}>"

    # CAUSES ==================================================================

    def disparity_matrix(self):
        df = pd.DataFrame(list(self._metadata_cache.run_parameters_values))
        df.index.name = "Iteration"
        df.columns.name = "Parameters"
        return df

    def changing_parameters(self):
        dm = self.disparity_matrix()
        uniques = dm.apply(np.unique)
        changes = uniques.apply(len) != 1
        changes.name = "Changes"
        return changes

    def coerce_parameter(self, parameter):
        if parameter is None:
            wpc = self.changing_parameters()
            candidates = wpc[wpc].index.to_numpy()
            candidates_len = len(candidates)
            if candidates_len != 1:
                raise ValueError(
                    "The value of 'parameter' is ambiguous since it has "
                    f"{candidates_len} candidates: {candidates}"
                )
            parameter = candidates[0]
        elif parameter not in self.run_parameters_:
            raise ValueError(f"Unknown run_parameter {parameter!r}")
        return parameter

    def causes_by_parameter(self, *, parameter=None):
        parameter = self.coerce_parameter(parameter)
        run_params_values = self._metadata_cache.run_parameters_values
        causes = self._metadata_cache.causes

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
        parameter = self.coerce_parameter(parameter)
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
        parameter = self.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        groups = cdf.groupby(("Parameters", parameter))
        report = groups.mean()

        report.index.name = "Disparity"

        report.columns = [parameter]
        report.columns.name = "Causes"

        return report

    def describe_causes(self, *, parameter=None):
        parameter = self.coerce_parameter(parameter)
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

    # BIAS ====================================================================
    def modes_variance_sum(self):
        varsum = self._metadata_cache.modes_variances_sum.copy()
        varsum.name = "VarSum"
        return varsum

    def coerce_mode(self, mode):
        if mode is None:
            # maybe two modes have exactly the same variance_sum
            # for this reason we dont use argmax that only return the first max
            modes_varsum = self.modes_variance_sum()
            maxvalue = modes_varsum.max()

            candidates = modes_varsum.index[modes_varsum == maxvalue]
            candidates_len = len(candidates)

            if candidates_len != 1:
                raise ValueError(
                    "The value of 'mode' is ambiguous since it has "
                    f"{candidates_len} candidates: {candidates}"
                )

            mode = candidates[0]

        elif mode not in self.modes_:
            raise ValueError(f"Unknown mode {mode!r}")

        return mode

    def bias(self, *, mode=None):
        mode = self.coerce_mode(mode)
