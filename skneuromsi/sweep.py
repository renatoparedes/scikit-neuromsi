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

from . import core
from .utils import numcompress

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default range of values for parameter sweeps.
DEFAULT_RANGE = 90 + np.arange(0, 20, 2)

# =============================================================================
# PARALLEL FUNCTIONS
# =============================================================================

def _run_report(*, idx, model, run_kws, seed, compression_params):
    """Run the model with given parameters and process the result.

    Parameters
    ----------
    idx : int
        Index of the run.
    model : object
        Model object to run the parameter sweep on.
    run_kws : dict
        Keyword arguments to pass to the model's `run` method.
    seed : int
        Seed for the random number generator.
    compression_params : tuple
        Compression parameters for joblib.dump.

    Returns
    -------
    object
        Compressed results from the run.
    """
    model.set_random(np.random.default_rng(seed))
    result = model.run(**run_kws)
    compressed = core.compress_ndresult(result)
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
        Model object to run the parameter sweep on.
    target : str
        Name of the parameter to sweep over.
    range : array-like, optional
        Range of values to sweep over. Default is `DEFAULT_RANGE`.
    repeat : int, optional
        Number of times to repeat each run. Default is 100.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    seed : int, optional
        Seed for the random number generator. Default is None.
    compression_params : tuple, optional
        Compression parameters for joblib.dump. Default is
        `skneuromsi.core.DEFAULT_COMPRESSION_PARAMS`.
    tqdm_cls : class, optional
        Class to use for progress bars. Default is `tqdm`.

    Attributes
    ----------
    model : object
        The model object.
    range : ndarray
        The range of values to sweep over.
    repeat : int
        The number of times to repeat each run.
    n_jobs : int
        The number of jobs to run in parallel.
    target : str
        The name of the parameter to sweep over.
    seed : int or None
        The seed for the random number generator.
    random_ : numpy.random.Generator
        The random number generator.
    compression_params : tuple
        The compression parameters for joblib.dump.

    Raises
    ------
    ValueError
        If `repeat` is less than 1.
    TypeError
        If the model's `run` method doesn't have the specified `target`
        parameter.

    """

    def __init__(
        self,
        model,
        target,
        *,
        range=None,
        repeat=2,
        n_jobs=1,
        seed=None,
        compression_params=core.DEFAULT_COMPRESSION_PARAMS,
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
        core.validate_compression_params(compression_params)

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
        self._compression_params = (
            compression_params
            if isinstance(compression_params, int)
            else tuple(compression_params)
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

    @property
    def compression_params(self):
        """The compression parameters for joblib.dump."""
        return self._compression_params

    def __repr__(self):
        cls_name = type(self).__name__
        model_name = type(self.model).__name__
        target = self._target
        repeat = self._repeat
        cp = self._compression_params
        return (
            f"<{cls_name} model={model_name!r} "
            f"target={target!r} repeat={repeat} compression_params={cp!r}>"
        )

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

        Raises
        ------
        TypeError
            If the target parameter is included in run_kws.

        """
        if self._target in run_kws:
            raise TypeError(
                f"Parameter '{self._target}' is under control of "
                f"{type(self)!r} instance"
            )

        # copy model and precision to easy write the code
        model, compression_params = self._model, self._compression_params

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

        # run the first iteration sequentially to check if the memory is
        # sufficient
        cit, rkw, rkw_seed = next(rkw_combs)
        first_result = _run_report(
            idx=cit,
            model=model,
            run_kws=rkw,
            seed=rkw_seed,
            compression_params=compression_params,
        )

        !!!!ddtype_tools.memory_impact(first_result)


        with joblib.Parallel(n_jobs=self._n_jobs) as Parallel:
            drun = joblib.delayed(_run_report)
            results = Parallel(
                drun(
                    idx=cit,
                    model=model,
                    run_kws=rkw,
                    seed=rkw_seed,
                    compression_params=compression_params,
                )
                for cit, rkw, rkw_seed in rkw_combs
            )

        # add the first iteration to the results
        results.insert(0, first_result)

        # aggregate all the processed results into a single object
        final_result = core.NDResultCollection("Sweep", results)

        return final_result