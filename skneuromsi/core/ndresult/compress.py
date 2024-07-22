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

"""Compress ndresult data and store it in a compressed format."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy
import dataclasses as dclss
import io

import joblib

import numpy as np

import pandas as pd

import xarray as xa

from ..utils import ddtype_tools
from .result import NDResult


# =============================================================================
# CONSTANTS
# =============================================================================

#: Default compression parameters. (see joblib.dump)
DEFAULT_COMPRESSION_PARAMS = ("zlib", 9)

#: Types that can be compressed.
_COMPRESS_TYPES = (np.ndarray, xa.DataArray, pd.DataFrame, pd.Series)


# =============================================================================
# CLASSES
# =============================================================================


@dclss.dataclass(slots=True, frozen=True, repr=False)
class CompressedNDResult:
    """
    A dataclass for storing compressed NDResults.

    Attributes
    ----------
    data : dict
        The compressed data.
    original_memory_usage : ddtype_tools.MemoryUsage
        Memory usage of the original NDResult.
    compressed_memory_usage : ddtype_tools.MemoryUsage
        Memory usage of the compressed NDResult.
    compressed_extra_keys : frozenset
        Keys of compressed extra data.
    """

    data: dict
    compressed_extra_keys: frozenset
    original_memory_usage: ddtype_tools.MemoryUsage
    compressed_memory_usage: ddtype_tools.MemoryUsage

    @property
    def compression(self):
        """Calculate the compression ratio.

        Returns
        -------
        float
            The compression ratio.
        """
        return (
            self.compressed_memory_usage.size / self.original_memory_usage.size
        )

    def __repr__(self):
        """Return a string representation of the CompressedNDResult object.

        Returns
        -------
        str
            String representation of the object.
        """
        cls_name = type(self).__name__

        compressed_size = self.compressed_memory_usage.hsize
        compression_percentage = self.compression * 100

        return (
            f"<{cls_name} {compressed_size!r} ({compression_percentage:.2f}%)>"
        )


# COMPRESSION =================================================================


def _compress(obj, compress_params):
    """Compress an object using joblib.

    Parameters
    ----------
    obj : object
        The object to compress.
    compress_params : tuple
        Compression parameters for joblib.dump.

    Returns
    -------
    bytes
        The compressed object.
    """
    stream = io.BytesIO()
    joblib.dump(obj, stream, compress=compress_params)
    return stream.getvalue()


def compress_ndresult(ndresult, compress_params=DEFAULT_COMPRESSION_PARAMS):
    """Compress an NDResult object.

    Parameters
    ----------
    ndresult : NDResult
        The NDResult object to compress.
    compress_params : tuple, optional
        Compression parameters for joblib.dump (default is
        DEFAULT_COMPRESSION_PARAMS).

    Returns
    -------
    CompressedNDResult
        The compressed NDResult object.

    Raises
    ------
    TypeError
        If the input is not an NDResult object.
    """
    if not isinstance(ndresult, NDResult):
        raise TypeError("Not an NDResult")

    ndresult_dict = ndresult.to_dict()

    # all the parts without compression
    compressed_ndresult_dict = {
        k: v for k, v in ndresult_dict.items() if k not in ("nddata", "extra")
    }

    # compress nddata
    compressed_ndresult_dict["nddata"] = _compress(
        ndresult_dict["nddata"], compress_params=compress_params
    )

    # compress extra
    compressed_extra = {}
    compressed_extra_keys = set()
    for k, v in ndresult_dict["extra"].items():

        if isinstance(v, _COMPRESS_TYPES):
            v = _compress(v, compress_params=compress_params)
            compressed_extra_keys.add(k)

        compressed_extra[k] = v

    compressed_ndresult_dict["extra"] = compressed_extra

    original_memory_usage = ddtype_tools.memory_usage(ndresult)
    compressed_memory_usage = ddtype_tools.memory_usage(
        compressed_ndresult_dict
    )

    return CompressedNDResult(
        data=compressed_ndresult_dict,
        compressed_extra_keys=frozenset(compressed_extra_keys),
        original_memory_usage=original_memory_usage,
        compressed_memory_usage=compressed_memory_usage,
    )


# DECOMPRESSION ===============================================================


def _decompress(compressed_bytes):
    """
    Decompress an object using joblib.

    This function takes a compressed object in bytes format and returns the
    decompressed object using joblib's load function.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed object in bytes format.

    Returns
    -------
    object
        The decompressed object.

    Notes
    -----
    This function uses io.BytesIO to create a file-like object in memory
    from the input bytes, which is then passed to joblib.load for
    decompression.

    Examples
    --------
    >>> compressed_data = b'...'  # Some compressed bytes
    >>> decompressed_obj = decompress(compressed_data)
    """
    stream = io.BytesIO(compressed_bytes)
    return joblib.load(stream)


def decompress_ndresult(compressed_ndresult):
    """Decompress an NDResult object.

    Parameters
    ----------
    compressed_ndresult : CompressedNDResult
        The compressed NDResult object to decompress.

    Returns
    -------
    NDResult
        The decompressed NDResult object.

    Raises
    ------
    TypeError
        If the input is not a CompressedNDResult object.
    """
    if not isinstance(compressed_ndresult, CompressedNDResult):
        raise TypeError("Not a compressed NDResult")

    ndresult_dict = copy.deepcopy(compressed_ndresult.data)

    # uncompress data
    ndresult_dict["nddata"] = _decompress(ndresult_dict["nddata"])

    # uncompress extra
    extra = ndresult_dict["extra"]
    for k in compressed_ndresult.compressed_extra_keys:
        extra[k] = _decompress(extra[k])

    return NDResult(**ndresult_dict)
