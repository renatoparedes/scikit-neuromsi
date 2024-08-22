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
from skneuromsi.core.ndresult import result

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
        "mode1": [
            4,
        ],
    }
    with pytest.raises(ValueError):
        result.modes_to_data_array(modes_dict, None)

    # diff number of positions but one is empty
    modes_dict = {
        "mode0": [],
        "mode1": [
            4,
        ],
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


# def test_NDResult_creation():
#     modes_dict = {
#         "mode0": [[1, 2, 3], [4, 5, 6]],
#         "output": [[1, 2, 3], [4, 5, 6]],
#     }
#     ndres = result.NDResult.from_modes_dict(
#         mname="Model",
#         modes_dict=modes_dict,
#         mtype="Test",
#         output_mode="output",
#         nmap={},
#         time_range=(0, 1),
#         position_range=(0, 1),
#         time_res=0.01,
#         position_res=0.01,
#         causes=1,
#         run_params={},
#         extra={},
#         ensure_dtype=None,
#     )

#     assert isinstance(ndres, result.NDResult)
