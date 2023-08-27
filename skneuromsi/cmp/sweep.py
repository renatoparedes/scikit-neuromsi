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

import inspect
import itertools as it

import joblib

import numpy as np

from tqdm.auto import tqdm

from .ndcollection import NDResultCollection
from ..utils import storages

# =============================================================================
# CONSTANTS
# =============================================================================


DEFAULT_RANGE = 90 + np.arange(0, 20, 2)


# =============================================================================
# CLASS
# =============================================================================


class ParameterSweep:
    def __init__(
        self,
        model,
        target,
        *,
        range=None,
        repeat=100,
        n_jobs=1,
        seed=None,
        tqdm_cls=tqdm,
        storage="directory",
        storage_kws=None,
    ):
        if repeat < 1:
            raise ValueError("'repeat must be >= 1'")

        self._model = model
        self._range = (
            DEFAULT_RANGE.copy() if range is None else np.asarray(range)
        )
        self._repeat = int(repeat)
        self._n_jobs = int(n_jobs)
        self._target = str(target)
        self._seed = seed
        self._random = np.random.default_rng(seed)
        self._tqdm_cls = tqdm_cls
        self._storage = storage
        self._storage_kws = {} if storage_kws is None else dict(storage_kws)

        run_signature = inspect.signature(model.run)
        if self._target not in run_signature.parameters:
            mdl_name = type(model).__name__
            raise TypeError(
                f"Model '{mdl_name}.run()' has no '{self._target}' parameter"
            )

    @property
    def model(self):
        return self._model

    @property
    def range(self):
        return self._range

    @property
    def repeat(self):
        return self._repeat

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def target(self):
        return self._target

    @property
    def seed(self):
        return self._seed

    @property
    def random_(self):
        return self._random

    @property
    def storage(self):
        return self._storage

    @property
    def storage_kws(self):
        return self._storage_kws.copy()

    def _run_kwargs_combinations(self, run_kws):
        iinfo = np.iinfo(int)

        def combs_gen():
            # combine all targets with all possible values
            tgt_x_range = it.product([self._target], self._range)
            current_iteration = 0
            for tgt_comb in tgt_x_range:
                # the combination as dict
                comb_as_kws = dict([tgt_comb])
                comb_as_kws.update(run_kws)

                # repeat the combination the number of times
                for _ in range(self._repeat):
                    seed = self._random.integers(low=0, high=iinfo.max)
                    yield current_iteration, comb_as_kws.copy(), seed
                    current_iteration += 1

        combs_size = len(self._range) * self._repeat

        return combs_gen(), combs_size

    def _run_report(self, idx, run_kws, seed, results):
        model = self._model
        model.set_random(np.random.default_rng(seed))
        results[idx] = model.run(**run_kws)

    def run(self, **run_kws):
        if self._target in run_kws:
            raise TypeError(
                f"Parameter '{self._target}' "
                f"are under control of {type(self)!r} instance"
            )

        # get all the configurations
        rkw_combs, runs_total = self._run_kwargs_combinations(run_kws)

        # if we need to add a progress bar we extract the iterable from it
        if self._tqdm_cls:
            rkw_combs = iter(
                self._tqdm_cls(
                    iterable=rkw_combs,
                    total=runs_total,
                    desc=f"Sweeping {self._target!r}",
                )
            )

        # Prepare to run
        storage_type = self._storage
        size = runs_total
        tag = type(self).__name__
        storage_kws = self._storage_kws

        with storages.storage(
            storage_type, size=size, tag=tag, **storage_kws
        ) as results:
            # execute the first iteration synchronous so if some configuration
            # fails we can catch it here
            cit, rkw, rkw_seed = next(rkw_combs)
            self._run_report(cit, rkw, rkw_seed, results)

            with joblib.Parallel(n_jobs=self._n_jobs) as Parallel:
                drun = joblib.delayed(self._run_report)
                Parallel(
                    drun(cit, rkw, rkw_seed, results)
                    for cit, rkw, rkw_seed in rkw_combs
                )

        result = NDResultCollection(tag, results, tqdm_cls=self._tqdm_cls)

        return result
