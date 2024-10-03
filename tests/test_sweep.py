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

"""Tests for skneuromsi/testing.py"""

# =============================================================================
# IMPORTS
# =============================================================================

from unittest import mock

import numpy as np

import psutil

import pytest

from skneuromsi.mle import AlaisBurr2004
from skneuromsi import core, sweep

# =============================================================================
# TESTS
# =============================================================================


def test_ParameterSweep(silenced_tqdm_cls):
    param_sweep = sweep.ParameterSweep(
        AlaisBurr2004(),
        target="auditory_position",
        repeat=1,
        tqdm_cls=silenced_tqdm_cls,
        seed=42,
    )

    assert isinstance(param_sweep.model, AlaisBurr2004)
    assert param_sweep.n_jobs is None
    assert (
        param_sweep.random_.bit_generator.state
        == np.random.default_rng(42).bit_generator.state
    )
    assert param_sweep.compression_params == core.DEFAULT_COMPRESSION_PARAMS
    assert param_sweep.tqdm_cls is silenced_tqdm_cls

    result = param_sweep.run()
    assert result.name == sweep.ParameterSweep.__name__
    assert len(result) == param_sweep.expected_result_length_
    assert result.coerce_parameter() == param_sweep.target


def test_ParameterSweep_repr():
    sweep_model = sweep.ParameterSweep(
        AlaisBurr2004(),
        target="auditory_position",
    )
    expected = (
        "<ParameterSweep model='AlaisBurr2004' "
        "target='auditory_position' repeat=2 compression_params=('lz4', 9)>"
    )
    assert repr(sweep_model) == expected


def test_ParameterSweep_repeat_lt_1():
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(), target="auditory_position", repeat=0
        )


def test_ParameterSweep_target_not_in_run():
    with pytest.raises(ValueError):
        sweep.ParameterSweep(AlaisBurr2004(), target="foo")


def test_ParameterSweep_mem_warning_and_error_in_run():
    # mem_warning must be between 0 and 1
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(), target="auditory_position", mem_warning_ratio=-1
        )
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(), target="auditory_position", mem_warning_ratio=2
        )

    # mem_error must be between 0 and 1
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(), target="auditory_position", mem_error_ratio=-1
        )
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(), target="auditory_position", mem_error_ratio=2
        )

    # mem_error must be >= than mem_warning
    with pytest.raises(ValueError):
        sweep.ParameterSweep(
            AlaisBurr2004(),
            target="auditory_position",
            mem_error_ratio=0.8,
            mem_warning_ratio=1,
        )


def test_ParameterSweep_run_parameter_under_sweep_control():
    param_sweep = sweep.ParameterSweep(
        AlaisBurr2004(),
        target="auditory_position",
    )

    with pytest.raises(ValueError):
        param_sweep.run(auditory_position=3)


def test_ParameterSweep_exeed_memory():
    param_sweep = sweep.ParameterSweep(
        AlaisBurr2004(),
        target="auditory_position",
        tqdm_cls=None,
        mem_warning_ratio=0.8,
        mem_error_ratio=1.0,
    )

    # get available memory
    available_mem = psutil.virtual_memory().total

    # each element is 10 times te total size of RAM, so it must fail
    each_result_mem_usage = available_mem * 10

    with mock.patch(
        "pympler.asizeof.asizeof", return_value=each_result_mem_usage
    ):
        with pytest.raises(sweep.ToBigForAvailableMemoryError):
            param_sweep.run()


def test_ParameterSweep_warning_memory():
    param_sweep = sweep.ParameterSweep(
        AlaisBurr2004(),
        target="auditory_position",
        tqdm_cls=None,
        mem_warning_ratio=0.8,
        mem_error_ratio=1.0,
    )

    # get available memory
    available_mem = psutil.virtual_memory().total

    # how much memory trigger the warning + 5% (to guarantee the trigger)
    warning_mem = int(available_mem * (param_sweep.mem_warning_ratio + 0.15))

    # how many elements do we expect?
    expected_result_length = param_sweep.expected_result_length_

    # the sumatory of all elements must be equal to the warning memory
    each_result_mem_usage = int(warning_mem / expected_result_length)

    # trigger the warning 90% os memory used
    with mock.patch(
        "pympler.asizeof.asizeof", return_value=each_result_mem_usage
    ):
        with pytest.warns(sweep.MaybeTooBigForAvailableMemoryWarning):
            param_sweep.run()
