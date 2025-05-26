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

"""Tests for the `skneuromsi.core.ndresult.compress` module."""

# =============================================================================
# IMPORTS
# =============================================================================

import re

import numpy as np

import pandas as pd

from pympler import asizeof

import pytest

import skneuromsi as sknmsi
from skneuromsi.core.ndresult import compress

import xarray as xa


# =============================================================================
# TESTS
# =============================================================================


def test_validate_compression_params():
    compress.validate_compression_params(compress.DEFAULT_COMPRESSION_PARAMS)

    for is_compressed in [True, False, None]:
        compress.validate_compression_params(is_compressed)

    for compression_level in range(9):
        compress.validate_compression_params(compression_level)

    for compressor in ["zlib", "gzip", "bz2", "xz", "lzma", "lz4"]:
        for compression_level in range(9):
            compress.validate_compression_params(
                (compressor, compression_level)
            )
        with pytest.raises(ValueError):
            compress.validate_compression_params((compressor, -1))
        with pytest.raises(ValueError):
            compress.validate_compression_params((compressor, 10))

    for invalid in [10, ("foo", 1), (1, 2, 3), -1]:
        with pytest.raises(ValueError):
            compress.validate_compression_params(invalid)


def test_compress_ndresult(random_ndresult):
    extra = {
        "int": 1,
        "ndarray": np.array([1, 2, 3]),
        "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "series": pd.Series([1, 2, 3]),
        "dataarray": xa.DataArray([1, 2, 3]),
    }

    original_ndres = random_ndresult(extra=extra, seed=42)
    original_ndres_dict = original_ndres.to_dict()
    original_memory_usage = asizeof.asizeof(original_ndres)

    comp_ndres = compress.compress_ndresult(original_ndres)
    comp_memory_usage = asizeof.asizeof(comp_ndres.data)

    # as a whole
    assert isinstance(comp_ndres, compress.CompressedNDResult)

    # same attrs
    assert comp_ndres.data.keys() == original_ndres_dict.keys()

    # data
    assert isinstance(comp_ndres.data["nddata"], bytes)

    # extra
    assert original_ndres.extra_.keys() == comp_ndres.data["extra"].keys()

    assert comp_ndres.compressed_extra_keys.symmetric_difference(extra) == {
        "int"
    }

    assert comp_ndres.data["extra"]["int"] == 1
    assert isinstance(comp_ndres.data["extra"]["int"], int)

    for key in comp_ndres.compressed_extra_keys:
        assert isinstance(comp_ndres.data["extra"][key], bytes)

    # not data and not extra
    for okey, ovalue in original_ndres_dict.items():
        if okey in ["extra", "nddata"]:
            continue
        cvalue = comp_ndres.data[okey]
        np.testing.assert_array_equal(ovalue, cvalue)

    # memory usage
    np.testing.assert_almost_equal(
        comp_ndres.compression_ratio,
        comp_memory_usage / original_memory_usage,
        decimal=2,
    )

    # repr
    pattern = r"<CompressedNDResult '\d+(\.\d+)? (kB|MB|GB|TB)' \(\d+\.\d+%\)>"
    comp_repr = repr(comp_ndres)
    assert re.match(pattern, comp_repr)


def test_decompress_ndresult(random_ndresult):
    extra = {
        "int": 1,
        "ndarray": np.array([1, 2, 3]),
        "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "series": pd.Series([1, 2, 3]),
        "dataarray": xa.DataArray([1, 2, 3]),
    }

    original_ndres = random_ndresult(extra=extra, seed=42)
    comp_ndres = compress.compress_ndresult(original_ndres)
    uncomp_ndres = compress.decompress_ndresult(comp_ndres)

    sknmsi.testing.assert_ndresult_allclose(original_ndres, uncomp_ndres)


def test_compress_ndresult_not_ndresult_instance():
    with pytest.raises(TypeError):
        compress.compress_ndresult(None)


def test_dcompress_ndresult_not_compressed_ndresult_instance():
    with pytest.raises(TypeError):
        compress.decompress_ndresult(None)
