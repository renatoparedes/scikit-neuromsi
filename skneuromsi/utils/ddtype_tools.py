#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# This code was ripped of from scikit-criteria on 2022-March-22.
# https://github.com/quatrope/scikit-criteria/blob/7f61c98/skcriteria/utils/accabc.py
# Util this point the copytight is

# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

""""""

# =============================================================================
# IMPORTS
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses as dclss
from typing import Iterable, Mapping

import humanize

import numpy as np

import pandas as pd

from pympler import asizeof

import xarray as xa

# =============================================================================
# CONSTANTS
# =============================================================================

#: Typess to inspect dtypes with .dtypes
_DTYPES_TYPES = (pd.DataFrame, xa.Dataset)

#: Typess to inspect dtypes with .dtype
_DTYPE_TYPES = (np.ndarray, pd.Series, xa.DataArray)

#: dtype casteable types
_ASTYPE_TYPES = _DTYPES_TYPES + _DTYPE_TYPES

#: Scalar iterables
_SCALAR_ITERABLES = (str,)


# =============================================================================
# CHECKS
# =============================================================================


def is_astypeable(obj):
    """Check if an object is astypeable."""
    return isinstance(obj, _ASTYPE_TYPES)


def is_class_astypeable(cls):
    """Check if a class is astypeable."""
    return issubclass(cls, _ASTYPE_TYPES)


def single_dtype(obj):
    """Check if an object is a single dtype."""
    return isinstance(obj, _DTYPE_TYPES)


def single_dtype_class(cls):
    """Chinsisinstance class is a single dtype class."""
    return issubclass(cls, _DTYPE_TYPES)


def multiple_dtype(obj):
    """Check if an object is a multiple dtype."""
    return isinstance(obj, _DTYPES_TYPES)


def multiple_dtype_class(cls):
    """Chinsisinstance class is a multiple dtype class."""
    return issubclass(cls, _DTYPE_TYPES)


# =============================================================================
# RECURSIVE ASTYPE
# =============================================================================


def deep_astype(obj, dtype=None):

    if dtype is None:
        return obj

    elif is_astypeable(obj):
        return obj.astype(dtype, copy=False)

    elif isinstance(obj, Mapping):
        original_type = type(obj)
        return original_type(
            {k: deep_astype(v, dtype) for k, v in obj.items()}
        )

    elif isinstance(obj, Iterable) and not isinstance(obj, _SCALAR_ITERABLES):
        original_type = type(obj)
        return original_type([deep_astype(e) for e in obj])

    return obj


# =============================================================================
# DEEP DTYPE INFO
# =============================================================================


@dclss.dataclass(frozen=True)
class _MemoryUsage:

    size: int

    @property
    def hsize(self):
        return humanize.naturalsize(self.size)

    def __repr__(self):
        return f"<MemoryUsage {self.hsize!r}>"


def _memory_usage(obj):
    size = asizeof.asizeof(obj)
    return _MemoryUsage(size=size)


def _deep_dtypes(obj, deep, max_deep, memory_usage):
    # Initialize the nested dtypes and the memory usage
    nested_dtypes = None
    memory = _memory_usage(obj) if memory_usage else None

    # if we exeed the deep, return its type and None for the data type
    if deep > max_deep:
        pass

    # Check if obj is has the .dtypes accessor
    elif isinstance(obj, _DTYPES_TYPES):
        nested_dtypes = obj.dtypes

    # Check if obj is has the .dtype accessor
    elif isinstance(obj, _DTYPE_TYPES):
        nested_dtypes = obj.dtype

    # If obj is a dictionary-like object
    elif isinstance(obj, Mapping):
        # Recursively call _deep_dtypes on each value
        # and return a dictionary with the keys and data types
        nested_dtypes = {
            k: _deep_dtypes(v, deep + 1, max_deep, memory_usage)
            for k, v in obj.items()
        }

    # If obj is an iterable (but not a dictionary-like and str object)
    elif isinstance(obj, Iterable) and not isinstance(obj, _SCALAR_ITERABLES):
        # Recursively call _deep_dtypes on each element
        # and return a list of tuples with (object type, data type)
        nested_dtypes = [
            _deep_dtypes(v, deep + 1, max_deep, memory_usage) for v in obj
        ]

    # If obj is not a scalar type, dictionary-like object, or iterable
    # return its type and None for the data type
    result = (
        (type(obj), nested_dtypes, memory)
        if memory_usage
        else (type(obj), nested_dtypes)
    )
    return result


def deep_dtypes(obj, *, root="root", max_deep=2, memory_usage=False):
    dict_obj = {root: obj}  # this add an extra level of nesting (level 0)

    # we extract the data types of the root object
    # and the root is ALWAYS a dict
    ddtypes = _deep_dtypes(
        dict_obj, deep=0, max_deep=max_deep, memory_usage=memory_usage
    )

    # remove the added dict of (dict, {root: obj}, [memusage])
    return ddtypes[1:] if memory_usage else ddtypes[1]
