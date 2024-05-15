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

"""test for skneuromsi.utils.numcompress"""


# =============================================================================
# IMPORTS
# =============================================================================

import random
import sys

import numpy as np

import pytest

import xarray as xa

from skneuromsi.utils import numcompress

# =============================================================================
# TESTS
# =============================================================================


def test_numcompress_Separators():
    for k, v in vars(numcompress.Separators).items():
        if k.startswith("_"):
            continue
        if isinstance(v, str) and ord(v) > 63:
            raise ValueError(f"Invalid separator {k}={v}, {v} must be < 63")


def test_numcompress_compress_decompress_works_for_single_int():
    series = [12345]
    text = "?qbW"
    assert numcompress.compress(series, 0) == text
    assert numcompress.decompress(text) == series


def test_numcompress_compress_decompress_works_in_lossy_fashion_for_longer_floats_when_enough_precision_not_specified():  # noqa: E501
    original_series = [12365.54524354, 14789.54699, 11367.67845123]
    lossy_series = [12365.545, 14789.547, 11367.678]
    text = "BqmvqVck}rCxizoE"
    assert numcompress.compress(original_series) == text
    assert numcompress.decompress(text) == lossy_series


def test_numcompress_compress_decompress_works_in_lossless_fashion_for_longer_floats_when_appropriate_precision_is_specified():  # noqa: E501
    original_series = [12365.54524354, 14789.54673, 11367.67845987]
    text = "Io{pcifu|_Folwenp}ak@f~itlgxf}@"
    assert numcompress.compress(original_series, 10) == text
    assert numcompress.decompress(text) == original_series


def test_numcompress_compression_ratio_for_series_of_epoch_timestamps():
    seed = 946684800  # start of year 2000
    series = [seed]
    previous = seed

    for _ in range(10000):
        current = previous + random.randrange(0, 600)
        series.append(current)
        previous = current

    original_size = sum(sys.getsizeof(i) for i in series)
    text = numcompress.compress(series, 0)
    compressed_size = sys.getsizeof(text)
    # reduction = ((original_size - compressed_size) * 100.0) / original_size

    assert compressed_size < original_size
    # print("10k timestamps compressed by ", round(reduction, 2), "%")
    assert numcompress.decompress(text) == series


def test_numcompress_compression_ratio_for_series_of_floats():
    seed = 1247.53
    series = [seed]
    previous = seed

    for _ in range(10000):
        current = previous + random.randrange(1000, 100000) * (10**-2)
        series.append(round(current, 3))
        previous = current

    original_size = sum(sys.getsizeof(i) for i in series)
    text = numcompress.compress(series)
    compressed_size = sys.getsizeof(text)
    # reduction = ((original_size - compressed_size) * 100.0) / original_size

    assert compressed_size < original_size
    # print("10k floats compressed by ", round(reduction, 2), "%")
    assert numcompress.decompress(text) == series


def test_numcompress_compress_none_value_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress(None)


def test_numcompress_compress_non_list_value_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress(23)


def test_numcompress_compress_decompress_works_with_empty_list():
    assert numcompress.compress([]) == ""
    assert numcompress.decompress("") == []


def test_numcompress_compress_non_numerical_list_value_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress([123, "someText", 456])


def test_numcompress_compress_non_integer_precision_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress([123, 125], precision="someValue")


def test_numcompress_compress_negative_precision_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress([123, 125], precision=-2)


def test_numcompress_compress_higher_than_limit_precision_raises_exception():
    with pytest.raises(ValueError):
        numcompress.compress(23, precision=17)


def test_numcompress_decompress_non_text_input_raises_exception():
    with pytest.raises(ValueError):
        numcompress.decompress(23)


def test_numcompress_decompress_invalid_text_input_raises_exception():
    with pytest.raises(ValueError):
        numcompress.decompress("^fhfjelr;")


def test_numcompress_compress_decompress_works_with_numpy_array():
    series = np.random.randint(1, 100, 100).reshape(10, 10)
    result = numcompress.decompress_ndarray(
        numcompress.compress_ndarray(series)
    )
    np.testing.assert_array_equal(result, series)
