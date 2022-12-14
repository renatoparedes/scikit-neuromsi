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
import re

import joblib

import numpy as np

import pandas as pd

from tqdm.auto import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================


DEFAULT_RANGE = 90 + np.arange(0, 20, 2)


# =============================================================================
# CLASS
# =============================================================================


class UnityReport:
    def __init__(
        self,
        model,
        *,
        range=None,
        repeat=100,
        n_jobs=1,
        target_rx=re.compile(r".*_position$"),
        seed=None,
        progress_cls=tqdm,
    ):
        self._model = model
        self._range = (
            DEFAULT_RANGE.copy() if range is None else np.asarray(range)
        )
        self._repeat = int(repeat)
        self._n_jobs = int(n_jobs)
        self._target_rx = re.compile(target_rx)
        self._random = np.random.default_rng(seed)
        self._progress_cls = progress_cls

        run_signature = inspect.signature(model.run)
        self._run_params_targets = frozenset(
            param
            for param in run_signature.parameters
            if self._target_rx.match(param)
        )
        if not self._run_params_targets:
            mdl_name = type(model).__name__
            raise TypeError(
                f"Model '{mdl_name}.run()' has no parameters that "
                f"match the regex {self._target_rx.pattern!r}"
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
    def target_rx(self):
        return self._target_rx

    @property
    def run_params_targets_(self):
        return self._run_params_targets

    def _run_kwargs_combinations(self, run_kws):
        iinfo = np.iinfo(int)

        def combs_gen():
            # combine all targets with all possible values
            tgt_x_range = it.chain(
                it.product([tgt], self._range)
                for tgt in self._run_params_targets
            )
            for tgt_comb in it.product(*tgt_x_range):
                # the combination as dict
                comb_as_kws = dict(tgt_comb)
                comb_as_kws.update(run_kws)

                # repeat the combination the number of times
                for _ in range(self._repeat):
                    seed = self._random.integers(low=0, high=iinfo.max)
                    yield comb_as_kws.copy(), seed

        combs_size = (
            (len(self._run_params_targets) * len(self._range))
        ) * self._repeat

        return combs_gen(), combs_size

    def _run_report(self, run_kws, seed):

        model = self._model
        model.set_random(np.random.default_rng(seed))

        response = model.run(**run_kws)
        row = {
            k: v for k, v in run_kws.items() if k in self._run_params_targets
        }
        row["causes"] = response.causes_
        return row

    def run(self, **run_kws):
        forbidden = self._run_params_targets.intersection(run_kws)
        if forbidden:
            raise TypeError(
                f"The parameter/s {set(forbidden)} "
                f"are under control of {type(self)!r} instance"
            )

        # get all the configurations
        rkw_combs, rkws_comb_len = self._run_kwargs_combinations(run_kws)
        if self._progress_cls:
            rkw_combs = iter(
                self._progress_cls(iterable=rkw_combs, total=rkws_comb_len)
            )
        # we execute the first iteration synchronous so if some configuration
        # fails we can catch it here
        rkw, rkw_seed = next(rkw_combs)
        first_response = self._run_report(rkw, rkw_seed)

        with joblib.Parallel(n_jobs=self._n_jobs) as P:
            drun = joblib.delayed(self._run_report)
            responses = P(drun(rkw, rkw_seed) for rkw, rkw_seed in rkw_combs)

        responses.insert(0, first_response)
        return pd.DataFrame(responses)
