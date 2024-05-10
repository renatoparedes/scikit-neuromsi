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

"""Module for performing parameter sweeps."""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect
import itertools as it

import joblib

import numpy as np

from tqdm.auto import tqdm

from .utils import storages

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default range of values for parameter sweeps.
DEFAULT_RANGE = 90 + np.arange(0, 20, 2)


# =============================================================================
# SWEEP VISITOR
# =============================================================================

class Visitor:

    def setup(self):
        ...

    def teardown(self):
        ...

    def process(self, result):
        return result

    def reduce(self, results):
        return results

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
        """The model object."""
        return self._model

    @property
    def range(self):
        """The range of values to sweep over."""
        return self._range

    @property
    def repeat(self):
        """The number of times to repeat the run for each value in the range."""
        return self._repeat

    @property
    def n_jobs(self):
        """The number of jobs to run in parallel."""
        return self._n_jobs

    @property
    def target(self):
        """The name of the parameter to sweep over."""
        return self._target

    @property
    def seed(self):
        """The seed for the random number generator."""
        return self._seed

    @property
    def random_(self):
        """The random number generator."""
        return self._random

    @property
    def storage(self):
        """The type of storage to use for storing the results."""
        return self._storage

    @property
    def storage_kws(self):
        """Additional keyword arguments to pass to the storage."""
        return self._storage_kws.copy()

    def _run_kwargs_combinations(self, run_kws):
        """Generate combinations of parameter values and seeds for the runs.

        Parameters
        ----------
        run_kws : dict
            Additional keyword arguments to pass to the model's `run` method.

        Returns
        -------
        generator
            A generator that yields tuples of (iteration, kwargs, seed) for
            each run.
        int
            The total number of runs.
        """
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
        """Run the model with the given parameters and store the results.

        Parameters
        ----------
        idx : int
            The index of the run.
        run_kws : dict
            The keyword arguments to pass to the model's `run` method.
        seed : int
            The seed for the random number generator.
        results : NDResultCollection
            The collection to store the results in.
        """
        model = self._model
        model.set_random(np.random.default_rng(seed))
        results[idx] = model.run(**run_kws)

    def run(self, **run_kws):
        """Run the sweep over the range of values for the target parameter.

        Parameters
        ----------
        **run_kws
            Additional keyword arguments to pass to the model's `run` method,
            except the target parameter.

        Returns
        -------
        NDResultCollection
            A collection of the results from the parameter sweep.

        """
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

        # creamos la clase visitor que se va a encargar de
        # gestionar como procesar los resultados
        with self._visitor_cls() as visitor:
            with joblib.Parallel(n_jobs=self._n_jobs) as Parallel:
                drun = joblib.delayed(self._run_report)
                results = Parallel(
                    drun(cit, rkw, rkw_seed)
                    for cit, rkw, rkw_seed in rkw_combs
                )

        result = visitor.reduce(results)

        return result