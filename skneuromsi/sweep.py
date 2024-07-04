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

import abc
import inspect
import itertools as it
import contextlib

import joblib

import numpy as np

from tqdm.auto import tqdm

import xarray as xa

from .core import NDResultCollection, compress_ndresult, decompress_ndresult

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default range of values for parameter sweeps.
DEFAULT_RANGE = 90 + np.arange(0, 20, 2)


# =============================================================================
# PARALLEL FUNCTIONS
# =============================================================================


def _run_report(*, idx, model, run_kws, seed):
    """Run the model with the given parameters and process the result \
        using the strategy.

        Parameters
        ----------
        idx : int
            The index of the run.
        run_kws : dict
            The keyword arguments to pass to the model's `run` method.
        seed : int
            The seed for the random number generator.
        strategy : SweepStrategyABC
            The sweep strategy to use for processing the result.
        """
    model.set_random(np.random.default_rng(seed))
    result = model.run(**run_kws)
    compressed = compress_ndresult(result)
    del result
    return compressed


# =============================================================================
# PARAMETER SWEEP
# =============================================================================


class ParameterSweep:
    """Perform a parameter sweep over a range of values for a target parameter.

    Parameters
    ----------
    model : object
        The model object to run the parameter sweep on.
    target : str
        The name of the parameter to sweep over.
    range : array-like, optional
        The range of values to sweep over. Defaults to `DEFAULT_RANGE`.
    repeat : int, optional
        The number of times to repeat each run. Defaults to 100.
    n_jobs : int, optional
        The number of jobs to run in parallel. Defaults to 1.
    seed : int, optional
        The seed for the random number generator. Defaults to None.
    strategy_cls : class, optional
        The class to use for the sweep strategy. Defaults to
        `DefaultSweepStrategy`.
    tqdm_cls : class, optional
        The class to use for progress bars. Defaults to `tqdm`.

    """

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
    ):
        # VALIDATIONS =========================================================
        if repeat < 1:
            raise ValueError("'repeat' must be >= 1")

        run_signature = inspect.signature(model.run)
        if str(target) not in run_signature.parameters:
            mdl_name = type(model).__name__
            raise TypeError(
                f"Model '{mdl_name}.run()' has no '{target}' parameter"
            )

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
        """The number of times to repeat each run."""
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

    def __repr__(self):
        cls_name = type(self).__name__
        model_name = type(self.model).__name__
        target = self._target
        repeat = self._repeat
        return f"<{cls_name} model={model_name!r} target={target!r} repeat={repeat}>"

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

    def run(self, **run_kws):
        """Run the sweep over the range of values for the target parameter.

        Parameters
        ----------
        **run_kws
            Additional keyword arguments to pass to the model's `run` method,
            except the target parameter.

        Returns
        -------
        object
            The aggregated results from all runs, as processed by the sweep
            strategy.

        """
        if self._target in run_kws:
            raise TypeError(
                f"Parameter '{self._target}' is under control of "
                f"{type(self)!r} instance"
            )

        # copy model to easy write
        model = self._model

        # get all the configurations
        rkw_combs, runs_total = self._run_kwargs_combinations(run_kws)

        # if we need to add a progress bar, we extract the iterable from it
        if self._tqdm_cls:
            rkw_combs = iter(
                self._tqdm_cls(
                    iterable=rkw_combs,
                    total=runs_total,
                    desc=f"Sweeping {self._target!r}",
                )
            )

        #
        with joblib.Parallel(n_jobs=self._n_jobs) as Parallel:
            drun = joblib.delayed(_run_report)
            results = Parallel(
                drun(idx=cit, model=model, run_kws=rkw, seed=rkw_seed)
                for cit, rkw, rkw_seed in rkw_combs
            )

        # aggregate all the processed results into a single object
        final_result = None  # strategy.aggregate_results(results)

        return final_result
