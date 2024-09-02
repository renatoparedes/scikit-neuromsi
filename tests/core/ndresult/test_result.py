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

"""Tests for the `skneuromsi.core.ndresult.compress` module.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import re

import numpy as np

import pandas as pd

from pympler import asizeof

import pytest

import skneuromsi as sknmsi
from skneuromsi.core import constants
from skneuromsi.core.ndresult import result, stats_acc, plot_acc

import xarray as xa


# =============================================================================
# TESTS
# =============================================================================


def test_modes_to_data_array():
    # no modes
    nddata = result.modes_to_data_array({}, None)

    assert nddata.ndim == 4
    assert nddata.size == 0
    assert len(nddata.coords) == 0
    np.testing.assert_equal(nddata.modes.to_numpy(), [0])
    np.testing.assert_equal(nddata.times.to_numpy(), [0])
    np.testing.assert_equal(nddata.positions.to_numpy(), [0])
    np.testing.assert_equal(nddata.positions_coordinates.to_numpy(), [])

    # two modes, one temporal dimension and 1 spatial dimension
    modes_dict = {"mode0": [1, 2, 3], "mode1": [4, 5, 6]}
    nddata = result.modes_to_data_array(modes_dict, None)

    assert nddata.ndim == 4
    assert nddata.size == 6
    assert len(nddata.coords) == 4
    np.testing.assert_equal(nddata.modes.to_numpy(), ["mode0", "mode1"])
    np.testing.assert_equal(nddata.times.to_numpy(), [0])
    np.testing.assert_equal(nddata.positions.to_numpy(), [0, 1, 2])
    np.testing.assert_equal(nddata.positions_coordinates.to_numpy(), ["x0"])

    # one mode, one temporal dimension and two spatial dimensions
    modes_dict = {"mode0": ([1, 2, 3], [4, 5, 6])}
    nddata = result.modes_to_data_array(modes_dict, None)

    assert nddata.ndim == 4
    assert nddata.size == 6
    assert len(nddata.coords) == 4
    np.testing.assert_equal(nddata.modes.to_numpy(), ["mode0"])
    np.testing.assert_equal(nddata.times.to_numpy(), [0])
    np.testing.assert_equal(nddata.positions.to_numpy(), [0, 1, 2])
    np.testing.assert_equal(
        nddata.positions_coordinates.to_numpy(), ["x0", "x1"]
    )

    # one mode two temporal dimensions one spatial dimension
    modes_dict = {"mode0": [[1, 2, 3], [4, 5, 6]]}
    nddata = result.modes_to_data_array(modes_dict, None)

    assert nddata.ndim == 4
    assert nddata.size == 6
    assert len(nddata.coords) == 4
    np.testing.assert_equal(nddata.modes.to_numpy(), ["mode0"])
    np.testing.assert_equal(nddata.times.to_numpy(), [0, 1])
    np.testing.assert_equal(nddata.positions.to_numpy(), [0, 1, 2])
    np.testing.assert_equal(nddata.positions_coordinates.to_numpy(), ["x0"])


def test_modes_to_data_array_invalid():

    # diff number of positions
    modes_dict = {
        "mode0": [1, 2, 3],
        "mode1": [4],
    }
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)

    # diff number of positions but one is empty
    modes_dict = {
        "mode0": [],
        "mode1": [4],
    }
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)

    # diff number of positions coordinates
    modes_dict = {"mode0": ([1, 2, 3], [4, 5, 6]), "mode1": [1, 2, 3]}
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)

    # diff number of times
    modes_dict = {"mode0": [[1, 2, 3], [4, 5, 6]], "mode1": [1, 2, 3]}
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)

    # try to confuse positions and times
    modes_dict = {
        "mode0": [[1, 2, 3], [4, 5, 6]],
        "mode1": ([1, 2, 3], [4, 5, 6]),
    }
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)


def test_NDResult_creation():
    modes_dict = {
        "mode0": [[1, 2, 3], [4, 5, 6]],
        "output": [[1, 2, 3], [4, 5, 6]],
    }
    ndres = result.NDResult.from_modes_dict(
        mname="Model",
        modes_dict=modes_dict,
        mtype="Test",
        output_mode="output",
        nmap={},
        time_range=(0, 1),
        position_range=(0, 3),
        time_res=0.5,
        position_res=1.0,
        causes=1,
        run_params={},
        extra={},
        ensure_dtype=float,
    )

    assert isinstance(ndres, result.NDResult)
    assert ndres.mname == "Model"
    assert ndres.mtype == "Test"
    assert ndres.output_mode == "output"
    assert ndres.nmap_ == {}
    np.testing.assert_array_equal(ndres.time_range, (0, 1))
    np.testing.assert_array_equal(ndres.position_range, (0, 3))
    assert ndres.time_res == 0.5
    assert ndres.position_res == 1.0
    assert ndres.causes_ == 1
    assert ndres.run_params == {}
    assert ndres.extra_ == {}
    np.testing.assert_array_equal(ndres.dims, constants.DIMENSIONS)
    np.testing.assert_array_equal(ndres.modes_, ["mode0", "output"])
    np.testing.assert_array_equal(ndres.times_, [0, 1])
    np.testing.assert_array_equal(ndres.positions_, [0, 1, 2])
    np.testing.assert_array_equal(ndres.positions_coordinates_, ["x0"])
    np.testing.assert_array_equal(
        ndres.to_xarray().to_numpy().flatten(), [1, 2, 3, 4, 5, 6] * 2
    )

    pattern = (
        r"<NDResult 'Model', modes=\['mode0' 'output'\], times=2, "
        r"positions=3, positions_coordinates=1, causes=1>"
    )
    assert re.match(pattern, repr(ndres))

    assert isinstance(ndres.plot, plot_acc.ResultPlotter)
    assert ndres.plot is ndres.plot
    assert isinstance(ndres.stats, stats_acc.ResultStatsAccessor)
    assert ndres.stats is ndres.stats


def test_NDResult_output_mode_not_found():
    modes_dict = {
        "mode0": [[1, 2, 3], [4, 5, 6]],
        "mode1": [[1, 2, 3], [4, 5, 6]],
    }
    with pytest.raises(ValueError, match="Output mode 'output' not found."):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )


def test_NDResult_lt_2_modes():
    modes_dict = {
        "output": [[1, 2, 3], [4, 5, 6]],
    }
    with pytest.raises(ValueError, match="At least two modes are required."):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )


def test_NDResult_invalid_time_range():
    modes_dict = {
        "mode0": [[1, 2, 3], [4, 5, 6]],
        "output": [[1, 2, 3], [4, 5, 6]],
    }
    with pytest.raises(
        ValueError,
        match=r"The time_range must be \(min, max\). Got \(0, 1, 2\)",
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1, 2),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )
    with pytest.raises(
        ValueError,
        match=r"The time_range must be \(min, max\). Got \(2, 1\)",
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(2, 1),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )


def test_NDResult_invalid_times():
    modes_dict = {
        "mode0": [[1, 2, 3]],
        "output": [[1, 2, 3]],
    }
    with pytest.raises(
        ValueError,
        match=(
            "The time_range and time_res do not match the data. "
            "Expected 2 times, got 1"
        ),
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )


def test_NDResult_invalid_position_range():
    modes_dict = {
        "mode0": [[1, 2, 3], [4, 5, 6]],
        "output": [[1, 2, 3], [4, 5, 6]],
    }
    with pytest.raises(
        ValueError,
        match=r"The position_range must be \(min, max\). Got \(0, 3, 4\)",
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1),
            position_range=(0, 3, 4),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )
    with pytest.raises(
        ValueError,
        match=r"The position_range must be \(min, max\). Got \(4, 3\)",
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(1, 2),
            position_range=(4, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )


def test_NDResult_invalid_positions():
    modes_dict = {
        "mode0": [[1, 2], [4, 5]],
        "output": [[1, 2], [4, 5]],
    }
    with pytest.raises(
        ValueError,
        match=(
            "The position_range and position_res do not match the data. "
            "Expected 3 positions, got 2"
        ),
    ):
        result.NDResult.from_modes_dict(
            mname="Model",
            modes_dict=modes_dict,
            mtype="Test",
            output_mode="output",
            nmap={},
            time_range=(0, 1),
            position_range=(0, 3),
            time_res=0.5,
            position_res=1.0,
            causes=1,
            run_params={},
            extra={},
            ensure_dtype=None,
        )
