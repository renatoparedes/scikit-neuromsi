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

import abc
import functools
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

import pandas as pd

import seaborn as sns

import xarray as xa

from .storages import make_storage
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


class NDResultCollection(Sequence):
    def __init__(self, ndresults, *, name=None, result_storage="memory"):
        self._name = str(name)
        self._len = len(ndresults)

        # resolve the result storage
        self._result_storage_type = result_storage

        mname_lst = []
        mtype_lst = []
        nmap_lst = []
        time_range_lst = []
        position_range_lst = []
        time_res_lst = []
        position_res_lst = []
        time_res_lst = []
        position_res_lst = []
        run_params_lst = []
        extra_lst = []
        causes_lst = []

        with make_storage(self._result_storage_type) as result_storage:
            for ndres in ndresults:
                mname_lst.append(ndres.mname)
                mtype_lst.append(ndres.mtype)
                nmap_lst.append(ndres.nmap_)
                time_range_lst.append(ndres.time_range)
                position_range_lst.append(ndres.position_range)
                time_res_lst.append(ndres.time_res)
                position_res_lst.append(ndres.position_res)
                run_params_lst.append(ndres.run_params.to_dict())
                extra_lst.append(ndres.extra_.to_dict())
                causes_lst.append(ndres.causes_)
                result_storage.add(ndres._nddata)

        self._mname_arr = np.array(mname_lst)
        self._mtype_arr = np.array(mtype_lst)
        self._nmap_arr = np.array(nmap_lst)
        self._time_range_arr = np.array(time_range_lst)
        self._position_range_arr = np.array(position_range_lst)
        self._time_res_arr = np.array(time_res_lst)
        self._position_res_arr = np.array(position_res_lst)
        self._time_res_arr = np.array(time_res_lst)
        self._position_res_arr = np.array(position_res_lst)
        self._run_params_arr = np.array(run_params_lst)
        self._extra_arr = np.array(extra_lst)
        self._causes_arr = np.array(causes_lst)

        self._result_storage = result_storage

    # Because is a Sequence ==================================================
    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._len

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            rargs = idxs.indices(len(self))
            idxs = range(*rargs)
        elif isinstance(idxs, (int, np.integer)):
            if idxs >= len(self):
                raise IndexError(
                    f"index {idxs} is out of bounds for "
                    f"collection with size {len(self)}"
                )
            idxs = [idxs]

        items = []
        for idx in idxs:
            item = NDResult(
                mname=self._mname_arr[idx],
                mtype=self._mtype_arr[idx],
                nmap=self._nmap_arr[idx],
                time_range=self._time_range_arr[idx],
                position_range=self._position_range_arr[idx],
                time_res=self._time_res_arr[idx],
                position_res=self._position_res_arr[idx],
                run_params=self._run_params_arr[idx],
                extra=self._extra_arr[idx],
                causes=self._causes_arr[idx],
                nddata=self._result_storage[idx].copy(),
            )
            items.append(item)

        if len(items) == 1:
            return items[0]

        return np.array(items, dtype=object)

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        return NDResultCollectionPlotter(self)

    @property
    def modes_(self):
        return self[0].modes_

    def __repr__(self):
        cls_name = type(self).__name__
        name = self.name
        length = len(self)
        stg = self._result_storage_type
        return f"<{cls_name} name={name!r}, len={length}, storage={stg!r}>"

    # CAUSES ==================================================================

    def disparity_matrix(self):
        df = pd.DataFrame(list(self._run_params_arr))
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

        columns = defaultdict(list)
        for rp, causes in zip(self._run_params_arr, self._causes_arr):

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
    def changing_mode(self):
        # ! darme cuenta de cual es el mode que se alterando si es None
        return "auditory"

    def bias(self, *, mode=None):
        mode = self.changing_mode()
