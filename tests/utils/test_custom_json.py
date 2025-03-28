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

"""Test module for the custom JSON encoding and decoding functionality.

This module contains test cases using pytest to verify the correct behavior
of the CustomJSONEncoder class and the dump(), dumps(), load(), and loads()
functions.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt

import numpy as np

import pytest

from skneuromsi.utils import custom_json as cjson


# =============================================================================
# TESTS
# =============================================================================
def test_custom_json_encoder():
    data = {
        "tuple": (1, 2, 3),
        "set": {1, 2, 3},
        "frozenset": frozenset([1, 2, 3]),
        "datetime": dt.datetime(2023, 5, 20, 10, 30, 0),
        "numpy_int": np.int32(42),
        "numpy_float": np.float64(3.14),
        "numpy_bool": np.bool_(True),
        "numpy_array": np.array([1, 2, 3]),
        "number": 1,
    }

    expected_json = (
        '{"tuple": [1, 2, 3], "set": [1, 2, 3], "frozenset": [1, 2, 3], '
        '"datetime": "2023-05-20T10:30:00", "numpy_int": 42, '
        '"numpy_float": 3.14, '
        '"numpy_bool": true, "numpy_array": [1, 2, 3], "number": 1}'
    )

    assert cjson.dumps(data) == expected_json


def test_custom_json_fail():
    with pytest.raises(TypeError):
        cjson.dumps(object())


def test_custom_json_dump_and_load(tmp_path):
    data = {"key": "value", "list": [1, 2, 3]}

    file_path = tmp_path / "test.json"
    with open(file_path, "w") as fp:
        cjson.dump(data, fp)

    with open(file_path, "r") as fp:
        loaded_data = cjson.load(fp)

    assert loaded_data == data


def test_custom_json_dumps_and_loads():
    data = {"key": "value", "list": [1, 2, 3]}

    json_string = cjson.dumps(data)
    loaded_data = cjson.loads(json_string)

    assert loaded_data == data
