#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module for performing parameter sweeps.

This module provides functionality to perform parameter sweeps over a range of
values for a target parameter in a given model. It includes classes for
different processing strategies and a main ParameterSweep class to orchestrate
the sweeps.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import abc
import inspect
import itertools as it
import warnings

import joblib

import numpy as np

from tqdm.auto import tqdm

from . import core, ndcollection
from .utils import doctools, memtools

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default range of values for parameter sweeps.
DEFAULT_RANGE = 90 + np.arange(0, 20, 2)

# =============================================================================
# ERRORS AND WARNINGS
# =============================================================================


class MaybeTooBigForAvailableMemoryWarning(UserWarning):
    """Warning raised when the result is potentially too big for \
    the available memory."""


class ToBigForAvailableMemoryError(MemoryError):
    """Error raised when the result is too big for the available memory."""


# =============================================================================
# PROCESSING STRATEGY
# =============================================================================


class ProcessingStrategyABC(abc.ABC):
    """Abstract base class for processing strategies.

    This class defines the interface for processing strategies used in
    parameter sweeps. Subclasses should implement the `map` and `reduce`
    methods.

    """

    @abc.abstractmethod
    def map(self, result):  # noqa: A003 "map" is shadowing a Python builtin
        """Process an individual result.

        Parameters
        ----------
        result : object
            The result to process.

        Returns
        -------
        object
            The processed result.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reduce(self, result_sequence, tag, tqdm_cls):
        """Combine a sequence of results into a single object.

        Parameters
        ----------
        result_sequence : iterable
            Sequence of results to combine.
        tag : str
            A tag to identify the results.
        tqdm_cls : class
            Progress bar class to use.

        Returns
        -------
        object
            The combined result.
        """
        raise NotImplementedError()

    def __repr__(self):
        """Return a string representation of the ProcessingStrategy object."""
        cls_name = type(self).__name__
        return cls_name


class NDCollectionProcessingStrategy(ProcessingStrategyABC):
    """Processing strategy for ND collections.

    This strategy compresses individual results and combines them into an
    NDResultCollection.

    Parameters
    ----------
    compression_params : tuple
        Compression parameters for joblib.dump. Defaults to
        core.DEFAULT_COMPRESSION_PARAMS.

    """

    def __init__(self, *, compression_params=None):
        compression_params = (
            core.DEFAULT_COMPRESSION_PARAMS
            if compression_params is None
            else compression_params
        )
        core.validate_compression_params(compression_params)
        self._compression_params = compression_params

    @doctools.doc_inherit(ProcessingStrategyABC.map)
    def map(self, result):  # noqa: A003 "map" is shadowing a Python builtin
        return core.compress_ndresult(
            result, compression_params=self._compression_params
        )

    @doctools.doc_inherit(ProcessingStrategyABC.reduce)
    def reduce(self, result_sequence, tag, tqdm_cls):
        return ndcollection.NDResultCollection(
            tag, result_sequence, tqdm_cls=tqdm_cls
        )


# =============================================================================
# PARALLEL FUNCTIONS
# =============================================================================


def _run_report(*, idx, model, run_kws, seed, processing_strategy):
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
    processing_strategy : ProcessingStrategy
        Processing strategy to use.

    Returns
    -------
    object
        The processing_strategy.map result.
    """
    model.set_random(np.random.default_rng(seed))
    result = model.run(**run_kws)
    return processing_strategy.map(result)


# =============================================================================
# PARAMETER SWEEP
# =============================================================================


class ParameterSweep:
    """Perform a parameter sweep over a range of values for a target parameter.

    This class orchestrates the parameter sweep process, including parallel
    execution of model runs and result aggregation.

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
    processing_strategy : ProcessingStrategy, optional
        Processing strategy to use. Default is
        `NDCollectionProcessingStrategy`.
    mem_warning_ratio : float, optional
        Ratio of available memory to trigger a warning. Default is 0.8.
    mem_error_ratio : float, optional
        Ratio of available memory to raise an error. Default is 1.0.
    tqdm_cls : class, optional
        Class to use for progress bars. Default is `tqdm`.

    Raises
    ------
    TypeError
        If the target parameter is not in the model's `run` method.
    ValueError
        If `repeat` is less than 1, mem_warning_ratio is not in [0, 1],
        mem_error_ratio is not in [0, 1], or the compression parameters are
        not valid.

    Notes
    -----
    The parameter sweep is performed in parallel using joblib.
    """

    def __init__(
        self,
        model,
        target,
        *,
        range=None,  # noqa: A002 "range" is shadowing a Python builtin
        repeat=2,
        n_jobs=None,
        seed=None,
        processing_strategy=None,
        mem_warning_ratio=0.8,
        mem_error_ratio=1.0,
        tqdm_cls=tqdm,
    ):
        # VALIDATIONS =========================================================
        if repeat < 1:
            raise ValueError("'repeat' must be >= 1")

        # check if the model has the target parameter in the run method
        run_signature = inspect.signature(model.run)
        if str(target) not in run_signature.parameters:
            mdl_name = type(model).__name__
            raise ValueError(
                f"Model '{mdl_name}.run()' has no '{target}' parameter"
            )

        # mem warning and error ratio
        if not (0 <= mem_warning_ratio <= 1):
            raise ValueError("'mem_warning_ratio' must be in [0, 1]")
        if not (0 <= mem_error_ratio <= 1):
            raise ValueError("'mem_error_ratio' must be in [0, 1]")
        if mem_warning_ratio > mem_error_ratio:
            raise ValueError(
                "'mem_warning_ratio' must be <= 'mem_error_ratio'"
            )

        self._model = model
        self._range = (
            DEFAULT_RANGE.copy() if range is None else np.asarray(range)
        )
        self._repeat = int(repeat)
        self._n_jobs = None if n_jobs is None else int(n_jobs)
        self._target = str(target)
        self._random = np.random.default_rng(seed)
        self._mem_warning_ratio = float(mem_warning_ratio)
        self._mem_error_ratio = float(mem_error_ratio)
        self._processing_strategy = (
            NDCollectionProcessingStrategy()
            if processing_strategy is None
            else processing_strategy
        )
        self._tqdm_cls = tqdm_cls

    @property
    def model(self):
        """The model object."""
        return self._model

    @property
    def range(self):  # noqa: A003 "range" is shadowing a Python builtin
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
    def random_(self):
        """The random number generator."""
        return self._random

    @property
    def expected_result_length_(self):
        """The expected length of the result."""
        return len(self.range) * self.repeat

    @property
    def processing_strategy(self):
        """The processing strategy."""
        return self._processing_strategy

    @property
    def tqdm_cls(self):
        """The class to use for progress bars."""
        return self._tqdm_cls

    @property
    def mem_warning_ratio(self):
        """The memory warning ratio."""
        return self._mem_warning_ratio

    @property
    def mem_error_ratio(self):
        """The memory error ratio."""
        return self._mem_error_ratio

    # REPRESENTATION ==========================================================
    def __repr__(self):
        """Return a string representation of the ParameterSweep object."""
        cls_name = type(self).__name__
        model_name = type(self.model).__name__
        target = self._target
        repeat = self._repeat
        ps = self._processing_strategy
        return (
            f"<{cls_name} model={model_name!r} "
            f"target={target!r} repeat={repeat} processing_strategy={ps!r}>"
        )

    # GENERATE ALL THE EXPERIMENT COMBINATIONS ================================
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

        return combs_gen()

    def _check_if_it_fit_in_memory(self, result, results_total):
        """Check if 'results_total' of the result fits in memory.

        Parameters
        ----------
        result : object
            A single result to check.
        results_total : int
            Total number of expected results.

        Raises
        ------
        ToBigForAvailableMemoryError
            If the result exceeds the available memory by the specified ratio.

        Warnings
        --------
        MaybeTooBigForAvailableMemoryWarning
            If the result is approaching the available memory limit.

        """
        memimpact = memtools.memory_impact(result, num_objects=results_total)
        if memimpact.total_ratio >= self.mem_error_ratio:
            total_perc = memimpact.total_ratio * 100
            mem_error_perc = self.mem_error_ratio * 100
            havailable_memory = memimpact.havailable_memory
            raise ToBigForAvailableMemoryError(
                f"Result is {total_perc:.2f}% "
                f"exceeding the {mem_error_perc:.2f}% of the "
                f"memory available, which is {havailable_memory!r}% "
            )
        if memimpact.total_ratio >= self.mem_warning_ratio:
            total_perc = memimpact.total_ratio * 100
            mem_warning_perc = self.mem_warning_ratio * 100
            havailable_memory = memimpact.havailable_memory
            warnings.warn(
                f"Result is {total_perc:.2f}% "
                f"exceeding the {mem_warning_perc:.2f}% of the "
                f"memory available, which is {havailable_memory!r}% ",
                category=MaybeTooBigForAvailableMemoryWarning,
            )

    def run(self, **run_kws):
        """Run the sweep over the range of values for the target parameter.

        This method performs the parameter sweep by running the model multiple
        times with different parameter values. It handles parallel execution,
        memory checks, and result aggregation.

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
        ValueError
            If the target parameter is included in run_kws.
        ToBigForAvailableMemoryError
            If the result exceeds the available memory by the specified ratio.

        Warnings
        --------
        MaybeTooBigForAvailableMemoryWarning
            If the result is approaching the available memory limit.

        Notes
        -----
        This method uses joblib for parallel execution of the model runs.
        It first runs a single iteration to check memory usage before
        proceeding with the full parameter sweep.

        """
        if self._target in run_kws:
            raise ValueError(
                f"Parameter '{self._target}' is under control of "
                f"{type(self)!r} instance"
            )

        # copy model and processing_strategy to easy write the code
        model, processing_strategy = self._model, self._processing_strategy

        # get all the configurations
        rkw_combs = self._run_kwargs_combinations(run_kws)
        runs_total = self.expected_result_length_

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
            processing_strategy=processing_strategy,
        )

        # check if the memory is sufficient
        self._check_if_it_fit_in_memory(first_result, runs_total)

        # run the rest of the iterations in parallel
        with joblib.Parallel(n_jobs=self._n_jobs) as Parallel:
            drun = joblib.delayed(_run_report)
            results = Parallel(
                drun(
                    idx=cit,
                    model=model,
                    run_kws=rkw,
                    seed=rkw_seed,
                    processing_strategy=processing_strategy,
                )
                for cit, rkw, rkw_seed in rkw_combs
            )

        # add the first iteration to the results
        results.insert(0, first_result)

        # aggregate all the processed results into a single object
        tag = type(self).__name__
        final_result = processing_strategy.reduce(
            results, tag=tag, tqdm_cls=self._tqdm_cls
        )

        return final_result
