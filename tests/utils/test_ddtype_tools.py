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

"""Tests for the `skneuromsi.utils.ddtype_tools` module."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from skneuromsi.utils import ddtype_tools

import xarray as xa

# =============================================================================
# TESTS CHECKS
# =============================================================================


def test_is_astypeable():
    """Test the `is_astypeable` function."""
    assert ddtype_tools.is_astypeable(pd.DataFrame())
    assert ddtype_tools.is_astypeable(xa.Dataset())
    assert ddtype_tools.is_astypeable(np.array([]))
    assert not ddtype_tools.is_astypeable(list())


def test_is_class_astypeable():
    """Test the `is_class_astypeable` function."""
    assert ddtype_tools.is_class_astypeable(pd.DataFrame)
    assert ddtype_tools.is_class_astypeable(xa.Dataset)
    assert ddtype_tools.is_class_astypeable(np.ndarray)
    assert not ddtype_tools.is_class_astypeable(list)


def test_single_dtype():
    """Test the `single_dtype` function."""
    assert ddtype_tools.single_dtype(np.array([]))
    assert ddtype_tools.single_dtype(pd.Series())
    assert ddtype_tools.single_dtype(xa.DataArray())
    assert not ddtype_tools.single_dtype(pd.DataFrame())


def test_single_dtype_class():
    """Test the `single_dtype_class` function."""
    assert ddtype_tools.single_dtype_class(np.ndarray)
    assert ddtype_tools.single_dtype_class(pd.Series)
    assert ddtype_tools.single_dtype_class(xa.DataArray)
    assert not ddtype_tools.single_dtype_class(pd.DataFrame)


def test_multiple_dtype():
    """Test the `multiple_dtype` function."""
    assert ddtype_tools.multiple_dtype(pd.DataFrame())
    assert ddtype_tools.multiple_dtype(xa.Dataset())
    assert not ddtype_tools.multiple_dtype(np.array([]))


def test_multiple_dtype_class():
    """Test the `multiple_dtype_class` function."""
    assert ddtype_tools.multiple_dtype_class(pd.DataFrame)
    assert ddtype_tools.multiple_dtype_class(xa.Dataset)
    assert not ddtype_tools.multiple_dtype_class(np.ndarray)


# =============================================================================
# TESTS ASTYPE
# =============================================================================


def test_deep_astype():
    """Test the `deep_astype` function."""

    no_astypeable = ddtype_tools.deep_astype(1, float)
    assert no_astypeable == 1 and isinstance(no_astypeable, int)

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    same_df = ddtype_tools.deep_astype(df, None)
    assert df is same_df

    df_float = ddtype_tools.deep_astype(df, float)
    assert df_float.dtypes.unique() == ["float64"]

    dict_df_float = ddtype_tools.deep_astype({"foo": df}, float)
    assert dict_df_float["foo"].dtypes.unique() == ["float64"]

    list_df_float = ddtype_tools.deep_astype([df], float)
    assert list_df_float[0].dtypes.unique() == ["float64"]


# =============================================================================
# TESTS DTYPES
# =============================================================================


def test_deep_dtypes():
    """Test the `deep_dtypes` function."""
    obj = {"a": np.array([1, 2, 3]), "b": pd.DataFrame({"A": [4, 5, 6]})}

    # max_deep 0
    dtypes_md0 = ddtype_tools.deep_dtypes(obj, max_deep=0)
    assert isinstance(dtypes_md0, dict)
    assert isinstance(dtypes_md0["root"], tuple)
    assert len(dtypes_md0["root"]) == 2
    assert dtypes_md0["root"][0] is dict
    assert dtypes_md0["root"][1] is None

    # no memory usage
    dtypes_dict = ddtype_tools.deep_dtypes(obj)
    assert isinstance(dtypes_dict, dict)
    assert isinstance(dtypes_dict["root"], tuple)
    assert len(dtypes_dict["root"]) == 2
    assert dtypes_dict["root"][0] is dict
    assert dtypes_dict["root"][1] is not None

    # iterable instead
    dtypes_list = ddtype_tools.deep_dtypes([obj])
    assert isinstance(dtypes_list, dict)
    assert isinstance(dtypes_list["root"], tuple)
    assert len(dtypes_list["root"]) == 2
    assert dtypes_list["root"][0] is list
    assert dtypes_list["root"][1] is not None

    # with memory_usage
    dtypes_mem = ddtype_tools.deep_dtypes(obj, memory_usage=True)
    assert isinstance(dtypes_mem, dict)
    assert isinstance(dtypes_mem["root"], tuple)
    assert len(dtypes_mem["root"]) == 3
    assert dtypes_mem["root"][0] is dict
    assert dtypes_mem["root"][1] is not None
