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

from typing import Iterable, Mapping

import numpy as np

import pandas as pd

import xarray as xa

from .constants import (
    DIMENSIONS,
    D_MODES,
    D_POSITIONS,
    D_POSITIONS_COORDINATES,
    D_TIMES,
    XA_NAME,
)
from .plot import ResultPlotter
from .stats import ResultStatsAccessor
from ..utils import Bunch

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#: The data types that can be type-casted.
_ASTYPE_TYPES = (np.ndarray, pd.DataFrame, pd.Series, xa.DataArray, xa.Dataset)


def recursive_mapping_astype(obj, dtype=None):
    """Ensure that the given object has the specified data type.



    Parameters
    ----------
    obj : object
        The object to be converted to the specified data type.
    dtype : data type, optional
        The desired data type for the object. If None, the object is returned
        as-is.

    Returns
    -------
    object
        The object with the specified data type. If the input object is a
        mapping (dict-like), a new mapping with the same keys is returned,
        with the values converted to the specified data type. If the input
        object is neither a mapping nor a type that supports the `astype`
        method, the same object is returned.

    Notes
    -----
    This function recursively converts the values of a mapping (dict-like)
    to the specified data type. If the input object is a type that supports
    the `astype` method (e.g., NumPy array), it is converted using that method.
    Otherwise, the same object is returned.

    """
    if dtype is None:
        return obj

    elif isinstance(obj, _ASTYPE_TYPES):
        return obj.astype(dtype, copy=False)

    elif isinstance(obj, Mapping):
        original_type, new_obj = type(obj), {}

        for map_key, map_value in obj.items():
            new_obj[map_key] = recursive_mapping_astype(map_value, dtype)

        return original_type(new_obj)

    return obj


# =============================================================================
# DEEP DTYPE INFO
# =============================================================================


def deep_dtypes(obj, *, root="root"):
    """Recursively determine the data types of an object or nested objects.

    Parameters
    ----------
    obj : object
        The object for which to determine the data types.
    root : str, optional
        The name of the root object, if applicable. Default is "root".

    Returns
    -------
    dict or list or tuple
        If `obj` is a pandas DataFrame, returns a tuple of
        (`type(obj)`, `obj.dtypes`), where `obj.dtypes` is a Series containing
        the data types of each column in the DataFrame.

        If `obj` is a dictionary-like object, returns a dictionary where the
        keys are the keys of `obj`, and the values are the data types of the
        corresponding values in `obj`.

        If `obj` is an iterable (but not a dictionary-like object), returns a
        list where each element is a tuple of (data type, object type) for the
        corresponding element in `obj`.

        If `obj` is a type that can be cast with `np.astype`, returns a tuple
        of (`type(obj)`, `obj.dtype`).

        If `obj` is any other type, returns a tuple of (`type(obj)`, `None`).

    Notes
    -----
    This function is useful for inspecting the data types of complex nested
    objects, such as those containing dictionaries, lists, NumPy arrays, pandas
    DataFrames, xarray objects, and other data structures.

    """

    # If root is provided, wrap obj in a dictionary with the given root key
    obj = obj if root is None else {root: obj}

    # Check if obj is a pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return type(obj), obj.dtypes

    # Check if obj is a scalar type that can be cast with np.astype
    elif isinstance(obj, _ASTYPE_TYPES):
        return type(obj), obj.dtype

    # If obj is a dictionary-like object
    elif isinstance(obj, Mapping):
        # Recursively call _deep_dtype_info_dict on each value
        # and return a dictionary with the keys and data types
        return {k: _deep_dtype_info_dict(v, root=None) for k, v in obj.items()}

    # If obj is an iterable (but not a dictionary-like object)
    elif isinstance(obj, Iterable):
        # Recursively call _deep_dtype_info_dict on each element
        # and return a list of tuples with (object type, data type)
        return [(type(v), _deep_dtype_info_dict(v, root=None)) for v in obj]

    # If obj is not a scalar type, dictionary-like object, or iterable
    # return its type and None for the data type
    return type(obj), None


