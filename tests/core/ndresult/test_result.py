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

import io
import re

import numpy as np

import pandas as pd

from pympler import asizeof

import pytest

import skneuromsi as sknmsi
from skneuromsi.core import constants
from skneuromsi.core.ndresult import result, stats_acc, plot_acc
from skneuromsi.utils import dict_cmp

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


def test_NDResult_get_modes():
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
    expected = pd.DataFrame(
        {
            "mode0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "output": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=pd.MultiIndex.from_product(
            [[0, 1], [0, 1, 2], ["x0"]],
            names=["times", "positions", "positions_coordinates"],
        ),
    )
    expected.columns.name = "modes"

    modes = ndres.get_modes()

    pd.testing.assert_frame_equal(modes, expected)
    with pytest.raises(ValueError, match="modes 'invalid' not found"):
        ndres.get_modes(include="invalid")


def test_NDResult_get_times():
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
    expected = pd.DataFrame(
        {
            0: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            1: [4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
        },
        index=pd.MultiIndex.from_product(
            [["mode0", "output"], [0, 1, 2], ["x0"]],
            names=["modes", "positions", "positions_coordinates"],
        ),
    )
    expected.columns.name = "times"

    times = ndres.get_times()
    pd.testing.assert_frame_equal(times, expected)

    times = ndres.get_times(include=[0, 1])
    pd.testing.assert_frame_equal(times, expected)

    with pytest.raises(ValueError, match="times 'invalid' not found"):
        ndres.get_times(include="invalid")


def test_NDResult_get_positions():
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
    expected = pd.DataFrame(
        {
            0: [1.0, 4.0, 1.0, 4.0],
            1: [2.0, 5.0, 2.0, 5.0],
            2: [3.0, 6.0, 3.0, 6.0],
        },
        index=pd.MultiIndex.from_product(
            [["mode0", "output"], [0, 1], ["x0"]],
            names=["modes", "times", "positions_coordinates"],
        ),
    )
    expected.columns.name = "positions"

    positions = ndres.get_positions()
    pd.testing.assert_frame_equal(positions, expected)

    positions = ndres.get_positions(include=[0, 1, 2])
    pd.testing.assert_frame_equal(positions, expected)

    with pytest.raises(ValueError, match="positions 'invalid' not found"):
        ndres.get_positions(include="invalid")


def test_NDResult_get_position_coordinates():
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
    expected = pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.MultiIndex.from_product(
            [["mode0", "output"], [0, 1], [0, 1, 2]],
            names=["modes", "times", "positions"],
        ),
    )
    expected.columns.name = "positions_coordinates"

    position_coordinates = ndres.get_positions_coordinates()
    pd.testing.assert_frame_equal(position_coordinates, expected)

    position_coordinates = ndres.get_positions_coordinates(include="x0")
    pd.testing.assert_frame_equal(position_coordinates, expected)

    # invalid
    with pytest.raises(
        ValueError, match="positions_coordinates 'invalid' not found"
    ):
        ndres.get_positions_coordinates(include="invalid")


def test_NDResult_to_xarray():
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

    expected = xa.DataArray(
        data=[
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
        ],
        coords={
            "modes": ["mode0", "output"],
            "times": [0, 1],
            "positions": [0, 1, 2],
            "positions_coordinates": ["x0"],
        },
        dims=["modes", "times", "positions", "positions_coordinates"],
    )

    as_da = ndres.to_xarray()
    xa.testing.assert_equal(expected, as_da)


def test_NDResult_to_dict():
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

    expected = {
        "mname": "Model",
        "mtype": "Test",
        "output_mode": "output",
        "nmap": {},
        "time_range": np.array([0.0, 1.0]),
        "position_range": np.array([0.0, 3.0]),
        "time_res": 0.5,
        "position_res": 1.0,
        "causes": 1,
        "run_params": {},
        "extra": {},
        "nddata": xa.DataArray(
            data=[
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            ],
            coords={
                "modes": ["mode0", "output"],
                "times": [0, 1],
                "positions": [0, 1, 2],
                "positions_coordinates": ["x0"],
            },
            dims=["modes", "times", "positions", "positions_coordinates"],
        ),
    }

    as_dict = ndres.to_dict()
    dict_cmp.dict_allclose(expected, as_dict)


def test_NDResult_to_ndr(random_ndresult):
    ndres = random_ndresult()

    oo_buffer = io.BytesIO()
    ndres.to_ndr(oo_buffer)
    oo_buffer.seek(0)

    func_buffer = io.BytesIO()
    sknmsi.store_ndresult(func_buffer, ndres)
    func_buffer.seek(0)

    restored_from_oo = sknmsi.read_ndr(oo_buffer)
    restored_from_func = sknmsi.read_ndr(func_buffer)

    sknmsi.testing.assert_ndresult_allclose(
        restored_from_func, restored_from_oo
    )


def test_NDResult_astype_compared_with_deep_dtypes(random_ndresult):
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

    expected = {
        "mname": (str, None),
        "mtype": (str, None),
        "output_mode": (str, None),
        "nmap": (dict, {}),
        "time_range": (np.ndarray, np.dtype("float64")),
        "position_range": (np.ndarray, np.dtype("float64")),
        "time_res": (float, None),
        "position_res": (float, None),
        "causes": (int, None),
        "run_params": (dict, {}),
        "extra": (dict, {}),
        "nddata": (xa.DataArray, np.dtype("float64")),
    }

    assert expected == ndres.deep_dtypes()

    int_ndres = ndres.astype(int)

    int_expected = {
        "mname": (str, None),
        "mtype": (str, None),
        "output_mode": (str, None),
        "nmap": (dict, {}),
        "time_range": (np.ndarray, np.dtype("int64")),
        "position_range": (np.ndarray, np.dtype("int64")),
        "time_res": (float, None),
        "position_res": (float, None),
        "causes": (int, None),
        "run_params": (dict, {}),
        "extra": (dict, {}),
        "nddata": (xa.DataArray, np.dtype("int64")),
    }

    assert int_expected == int_ndres.deep_dtypes()


def test_NDResult_dtypes(random_ndresult):
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

    expected = pd.DataFrame(
        {
            "Type": [
                str,
                str,
                str,
                dict,
                np.ndarray,
                np.ndarray,
                float,
                float,
                int,
                dict,
                dict,
                xa.core.dataarray.DataArray,
            ],
            "DType": [
                "-",
                "-",
                "-",
                "-",
                "float64",
                "float64",
                "-",
                "-",
                "-",
                "-",
                "-",
                "float64",
            ],
            "Size": [
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
            ],
        },
        index=[
            "mname",
            "mtype",
            "output_mode",
            "nmap",
            "time_range",
            "position_range",
            "time_res",
            "position_res",
            "causes",
            "run_params",
            "extra",
            "nddata",
        ],
    )

    expected.index.name = "Attribute"

    pd.testing.assert_frame_equal(expected, ndres.dtypes())
