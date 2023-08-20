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

from collections.abc import Sequence

import methodtools

import numpy as np

import pandas as pd

from tqdm.auto import tqdm

from ...core import constants as cons
from ...utils import Bunch
from . import plot_acc, bias_acc, causes_acc


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

    # all the run_parameters/modes are the same, so lets take the last one
    modes = ndres.modes_
    output_mode = ndres.output_mode
    run_parameters = tuple(ndres.run_params.to_dict())
    dims = ndres.dims

    # Resume the series collection into a single one we use sum instead of
    # numpy sum, because we want a pandas.Series and not a numpy array.
    # Also we assign the name to the Series.
    modes_variances_sum = sum(modes_variances)
    modes_variances_sum.name = "VarSum"

    cache = Bunch(
        "ndcollection_metadata_cache",
        {
            "modes": modes,
            "run_parameters": run_parameters,
            "mnames": np.asarray(mnames),
            "mtypes": np.asarray(mtypes),
            "output_mode": output_mode,
            "nmaps": np.asarray(nmaps),
            "time_ranges": np.asarray(time_ranges),
            "position_ranges": np.asarray(position_ranges),
            "time_resolutions": np.asarray(time_resolutions),
            "position_resolutions": np.asarray(position_resolutions),
            "run_parameters_values": np.asarray(run_parameters_values),
            "extras": np.asarray(extras),
            "causes": np.asarray(causes),
            "modes_variances_sum": modes_variances_sum,
            "dims": dims,
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

    @property
    def modes_(self):
        return self._metadata_cache.modes

    @property
    def output_mode_(self):
        return self._metadata_cache.output_mode

    @property
    def input_modes_(self):
        candidates = self.modes_
        return candidates[~(candidates == self.output_mode_)]

    @property
    def run_parameters_(self):
        return self._metadata_cache.run_parameters

    @property
    def dims_(self):
        return self._metadata_cache.dims

    def __repr__(self):
        cls_name = type(self).__name__
        name = self.name
        length = len(self)
        storage = type(self._ndresults).__name__
        return f"<{cls_name} {name!r} len={length} storage={storage!r}>"

    # PARAMETERS ANALYSIS =====================================================

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

    def modes_variance_sum(self):
        cache = self._metadata_cache
        varsum = cache.modes_variances_sum.copy()
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

    def coerce_dimension(self, prefer=None):
        """Coerce and validate the provided prefered dimension or select the \
        default dimension if None.

        If no dimension is provided, the method prefers to use the 'time'
        dimension; otherwise, it check if the provided dimension exists
        criteria.

        Parameters
        ----------
        prefer : str, optional
            The dimension to be coerced, or the default dimension if None is
            provided.

        Returns
        -------
        str
            The coerced or selected dimension.

        """
        if prefer is None:
            prefer = cons.D_TIMES
        elif prefer not in self.dims_:
            raise ValueError(f"Unknow dimension {prefer}")
        return prefer

    # ACCESORS ================================================================

    @methodtools.lru_cache(maxsize=None)
    @property
    def causes(self):
        return causes_acc.NDResultCausesAcc(self)

    @methodtools.lru_cache(maxsize=None)
    @property
    def bias(self):
        return bias_acc.NDResultBiasAcc(self)

    @methodtools.lru_cache(maxsize=None)
    @property
    def plot(self):
        return plot_acc.NDResultCollectionPlotter(self)
