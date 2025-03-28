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

"""Utilities for working with data types and memory usage in Python objects."""


# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Iterable, Mapping

import numpy as np

import pandas as pd

import xarray as xa

from . import memtools

# =============================================================================
# CONSTANTS
# =============================================================================

#: Typess to inspect dtypes with .dtypes
_MULTIPLE_DTYPES_TYPES = (pd.DataFrame, xa.Dataset)

#: Typess to inspect dtypes with .dtype
_SINGLE_DTYPE_TYPES = (np.ndarray, pd.Series, xa.DataArray)

#: dtype casteable types
_ASTYPE_TYPES = _MULTIPLE_DTYPES_TYPES + _SINGLE_DTYPE_TYPES

#: Scalar iterables
_SCALAR_ITERABLES = (str,)


# =============================================================================
# CHECKS
# =============================================================================


def is_astypeable(obj):
    """Check if an object support the message astype."""
    return isinstance(obj, _ASTYPE_TYPES)


def is_class_astypeable(cls):
    """Check if object of class support the message astype."""
    return issubclass(cls, _ASTYPE_TYPES)


def single_dtype(obj):
    """Check if an object has an attribute dtype."""
    return isinstance(obj, _SINGLE_DTYPE_TYPES)


def single_dtype_class(cls):
    """Check if object of class has an attribute dtype."""
    return cls in _SINGLE_DTYPE_TYPES


def multiple_dtype(obj):
    """Check if an object has an attribute dtypes."""
    return isinstance(obj, _MULTIPLE_DTYPES_TYPES)


def multiple_dtype_class(cls):
    """Check if object of class has an attribute dtypes."""
    return cls in _MULTIPLE_DTYPES_TYPES


# =============================================================================
# RECURSIVE ASTYPE
# =============================================================================


def deep_astype(obj, dtype=None):
    """Recursively cast the data type of an object and its nested objects.

    Parameters
    ----------
    obj : object
        The object to cast.
    dtype : data type, optional
        The data type to cast to. If None (default), the object is
        returned as is.

    Returns
    -------
    object
        The object with the new data type.

    """
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
        return original_type([deep_astype(e, dtype) for e in obj])

    return obj


# =============================================================================
# DEEP DTYPES
# =============================================================================


def _deep_dtypes(obj, deep, max_deep, memory_usage):
    """Recursively get the data types and memory usage of an object and its \
    nested objects.

    Parameters
    ----------
    obj : object
        The object to get the data types and memory usage for.
    deep : int
        The current depth of recursion.
    max_deep : int
        The maximum depth of recursion.
    memory_usage : bool
        Whether to calculate the memory usage.

    Returns
    -------
    tuple
        A tuple containing the type of the object, the nested data types,
        and optionally the memory usage.

    """
    # Initialize the nested dtypes and the memory usage
    nested_dtypes = None
    memory = memtools.memory_usage(obj) if memory_usage else None

    # if we exeed the deep, return its type and None for the data type
    if deep > max_deep:
        pass

    # Check if obj is has the .dtypes accessor
    elif isinstance(obj, _MULTIPLE_DTYPES_TYPES):
        nested_dtypes = obj.dtypes

    # Check if obj is has the .dtype accessor
    elif isinstance(obj, _SINGLE_DTYPE_TYPES):
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
    """Get the data types and optionally the memory usage of an object and \
    its nested objects.

    Parameters
    ----------
    obj : object
        The object to get the data types and memory usage for.
    root : str, optional
        The name of the root object (default is "root").
    max_deep : int, optional
        The maximum depth of recursion (default is 2).
    memory_usage : bool, optional
        Whether to calculate the memory usage (default is False).

    Returns
    -------
    tuple or dict
        If memory_usage is True, a dictionary with keys representing the
        nested objects and values representing their data types and
        memory usage.
        If memory_usage is False, a tuple with the nested data types.

    """
    dict_obj = {root: obj}  # this add an extra level of nesting (level 0)

    # we extract the data types of the root object
    # and the root is ALWAYS a dict
    ddtypes = _deep_dtypes(
        dict_obj, deep=0, max_deep=max_deep, memory_usage=memory_usage
    )

    # remove the added dict of (dict, {root: obj}, [memusage])
    return ddtypes[1]
