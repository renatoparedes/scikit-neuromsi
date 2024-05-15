#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# This code was extracted from numcompress on 13-may-2024.
# https://github.com/amit1rrr/numcompress/
# Until this point, the copyright and license are:
# License: MIT (https://opensource.org/licenses/MIT)
# Copyright (c) 2021 Amit Rathi
# All rights reserved.


# =============================================================================
# DOCS
# =============================================================================

"""numcompress - Simple compression and decompression of numerical series \
and NumPy arrays.

numcompress is a Python module that provides an easy way to compress and
decompress numerical series and NumPy arrays. It achieves compression ratios
above 80% and allows specifying the desired precision for floating-point
numbers (up to 10 decimal points). This module is particularly useful for
storing or transmitting stock prices, monitoring data, and other time series
data in a compressed string format.

The compression algorithm is based on the Google Encoded Polyline Format,
which has been modified to preserve arbitrary precision and apply it to any
numerical series. The development of this module was motivated by the
usefulness of the Time Aware Polyline built by Arjun Attam at HyperTrack.

Notes
-----

This is a forked version of the original numcompress library (version 0.1.2)
created by Amit Rathi. The original library provides a simple way to compress
and decompress numerical series and numpy arrays, achieving high compression
ratios. This forked version builds upon the original work and includes
additional features and improvements.

For more information about the original numcompress library, please refer to
the `original project repository <https://github.com/amit1rrr/numcompress>`_ .


References
----------

.. [RATHI2021] Rathi, A. (2021). numcompress: Simple way to compress and
   decompress numerical series & numpy arrays.
   https://github.com/amit1rrr/numcompress
.. [GOOGLEEPF] Google Developers. (n.d.). Encoded Polyline Algorithm Format.
   https://developers.google.com/maps/documentation/utilities/polylinealgorithm
.. [ATTAM2016] Attam, A. (2016, September 1). The Missing Dimension in
   Geospatial Data Formats. HyperTrack Blog.
   https://www.hypertrack.com/blog/2016/09/01/the-missing-dimension-in-geospatial-data-formats/


Examples
--------
>>> from numcompress import compress, decompress
>>> series = [145.7834, 127.5989, 135.2569]
>>> compressed = compress(series, precision=4)
>>> compressed
'Csi~wAhdbJgqtC'
>>> decompressed = decompress(compressed)
>>> decompressed
[145.7834, 127.5989, 135.2569]

>>> import numpy as np
>>> from numcompress import compress_ndarray, decompress_ndarray
>>> arr = np.random.randint(1, 100, 25).reshape(5, 5)
>>> compressed_arr = compress_ndarray(arr)
>>> decompressed_arr = decompress_ndarray(compressed_arr)
>>> np.array_equal(arr, decompressed_arr)
True

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

#: Lower and upper bounds for the precision parameter.
_PRECISION_LIMIT = 0, 10


class Separators:
    """Separators used in the compressed string representation.

    Should have ASCII value between (0, 63) to avoid overlapping with regular
    compress output.

    """

    #: Used for N-Dimensional array. Separates shape and the data.
    SHAPE = ","

    #: Separates dimension in the shape.
    DIMENSION = "*"


# =============================================================================
# PRIVATE
# =============================================================================


def _coerce_precision(precision):

    # type
    try:
        precision = int(precision)
    except ValueError:
        raise ValueError("Precision parameter needs to be a number.")

    # check bounds
    lower, upper = _PRECISION_LIMIT
    in_bound = lower <= precision <= upper
    if not in_bound:
        raise ValueError(
            f"Precision must be between {lower} to {upper} decimal places. "
            f"Found {precision}"
        )

    return precision


def _decompress_number(text, index):
    """Helper function to decompress a single number from the compressed \
    string.

    Parameters
    ----------
    text : str
        The compressed string representation.
    index : int
        The current index in the string.

    Returns
    -------
    tuple
        A tuple containing the updated index and the decompressed number.

    """
    result = 1
    shift = 0

    while True:
        b = ord(text[index]) - 63 - 1
        index += 1
        result += b << shift
        shift += 5

        if b < 0x1F:
            break

    return index, (~result >> 1) if (result & 1) != 0 else (result >> 1)


# =============================================================================
# LIST
# =============================================================================


def compress(series, precision=3):
    """
    Compress a list of numbers into a string representation.

    Parameters
    ----------
    series : list
        The list of numbers to be compressed.
    precision : int, optional
        The number of decimal places to preserve in the compressed
        representation. Default is 3.

    Returns
    -------
    str
        The compressed string representation of the input series.

    Raises
    ------
    ValueError
        If the input is not a list, precision is not an integer, precision is
        out of range, or if the series contains non-numeric values.
    """

    if not isinstance(series, list):
        raise ValueError("Input to compress should be of type list.")

    precision = _coerce_precision(precision)

    if not len(series):
        return ""

    # Store precision value at the beginning of the compressed text
    result = chr(precision + 63)

    # Store the last number in the series
    last_num = 0

    for num in series:
        if not isinstance(num, (int, float)):
            raise ValueError(
                "All input list items should either be of type int or float."
            )

        diff = num - last_num
        diff = int(round(diff * (10**precision)))
        diff = ~(diff << 1) if diff < 0 else diff << 1

        while diff >= 0x20:
            result += chr((0x20 | (diff & 0x1F)) + 63)
            diff >>= 5

        result += chr(diff + 63)
        last_num = num

    return result


def decompress(text):
    """Decompress a string representation of a series of numbers back into a \
    list of numbers.

    Parameters
    ----------
    text : str
        The compressed string representation to be decompressed.

    Returns
    -------
    list
        The decompressed list of numbers.

    Raises
    ------
    ValueError
        If the input is not a string or if the string is invalid or inaccurate.

    """
    result = []
    index = last_num = 0

    if not isinstance(text, str):
        raise ValueError("Input to decompress should be of type str.")

    if not text:
        return result

    # Decode precision value
    precision = ord(text[index]) - 63
    index += 1

    try:
        precision = _coerce_precision(precision)
    except ValueError:
        raise ValueError(
            "Invalid string sent to decompress. "
            "Please check the string for accuracy."
        )

    while index < len(text):
        index, diff = _decompress_number(text, index)
        last_num += diff
        result.append(last_num)

    result = [round(item * (10 ** (-precision)), precision) for item in result]
    return result


# =============================================================================
# NDARRAY
# =============================================================================


def compress_ndarray(series, precision=3):
    """Compress a NumPy ndarray into a string representation.

    Parameters
    ----------
    series : numpy.ndarray
        The NumPy ndarray to be compressed.
    precision : int, optional
        The number of decimal places to preserve in the compressed
        representation. Default is 3.

    Returns
    -------
    str
        The compressed string representation of the input ndarray.
    """
    shape = Separators.DIMENSION.join(map(str, series.shape))
    series_compressed = compress(series.flatten().tolist(), precision)
    return f"{shape}{Separators.SHAPE}{series_compressed}"


def decompress_ndarray(text):
    """Decompress a string representation of a NumPy ndarray back into an \
    ndarray.

    Parameters
    ----------
    text : str
        The compressed string representation to be decompressed.

    Returns
    -------
    numpy.ndarray
        The decompressed NumPy ndarray.
    """
    shape_str, series_text = text.rsplit(Separators.SHAPE)
    shape = tuple(
        int(dimension) for dimension in shape_str.split(Separators.DIMENSION)
    )
    series = decompress(series_text)
    return np.array(series).reshape(*shape)
