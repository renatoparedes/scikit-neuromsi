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
"""Public testing utility functions.

This module exposes "assert" functions which facilitate the comparison in a
testing environment of objects created in "scikit-neuromsi".

The functionalities are extensions of those present in "xarray.testing" and
"numpy.testing".

"""


# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np

import xarray as xa

from . import core, ndcollection
from .utils import dict_cmp


# =============================================================================
# ASSERTS
# =============================================================================


def _assert(cond, err_msg):
    """Asserts that a condition is true, otherwise raises an AssertionError \
    with a specified error message.

    This function exists to prevent asserts from being turned off with a
    "python -O."

    Parameters
    ----------
    cond : bool
        The condition to be evaluated.
    err_msg : str
        The error message to be raised if the condition is false.

    """
    if not cond:
        raise AssertionError(err_msg)


def assert_ndresult_allclose(
    left, right, rtol=1e-05, atol=1e-08, equal_nan=True, decode_bytes=True
):
    """Assert that two NDResult objects are approximately equal.

    Parameters
    ----------
    left : NDResult
        The first NDResult object to compare.
    right : NDResult
        The second NDResult object to compare.
    rtol : float, optional
        The relative tolerance parameter for the `assert_allclose` function
        (default is 1e-05).
    atol : float, optional
        The absolute tolerance parameter for the `assert_allclose` function
        (default is 1e-08).
    equal_nan : bool, optional
        Whether to compare NaN values in the arrays (default is True).
    decode_bytes : bool, optional
        Whether to decode bytes in the arrays (default is True).
        (See `xarray.testing.assert_allclose` for details).

    """
    _assert(isinstance(left, core.NDResult), "left is not an NDResult")

    if left is right:
        return

    _assert(isinstance(right, core.NDResult), "right is not an NDResult")

    _assert(left.mname == right.mname, "mname mismatch")
    _assert(left.mtype == right.mtype, "mtype mismatch")
    _assert(left.output_mode == right.output_mode, "output_mode mismatch")
    _assert(left.causes_ == right.causes_, "causes mismatch")
    _assert(left.time_res == right.time_res, "time_res mismatch")
    _assert(left.position_res == right.position_res, "position_res mismatch")

    np.testing.assert_allclose(
        left.time_range,
        right.time_range,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    np.testing.assert_allclose(
        left.position_range,
        right.position_range,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )

    # dicts
    assert dict_cmp.dict_allclose(
        left.nmap_, right.nmap_, rtol=rtol, atol=atol, equal_nan=equal_nan
    ), "nmap mismatch"

    assert dict_cmp.dict_allclose(
        left.run_parameters,
        right.run_parameters,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    ), "run_parameters mismatch"

    assert dict_cmp.dict_allclose(
        left.extra_, right.extra_, rtol=rtol, atol=atol, equal_nan=equal_nan
    ), "extra mismatch"

    xa.testing.assert_allclose(
        left.to_xarray(),
        right.to_xarray(),
        rtol=rtol,
        atol=atol,
        decode_bytes=decode_bytes,
    )


def assert_ndresult_collection_allclose(
    left, right, rtol=1e-05, atol=1e-08, equal_nan=True, decode_bytes=True
):
    """Assert that two NDResultCollection objects are approximately equal.

    Parameters
    ----------
    left : NDResultCollection
        The first NDResultCollection object to compare.
    right : NDResultCollection
        The second NDResultCollection object to compare.
    rtol : float, optional
        The relative tolerance parameter for the `assert_allclose` function
        (default is 1e-05).
    atol : float, optional
        The absolute tolerance parameter for the `assert_allclose` function
        (default is 1e-08).
    equal_nan : bool, optional
        Whether to compare NaN values in the arrays (default is True).
    decode_bytes : bool, optional
        Whether to decode bytes in the arrays (default is True).
        (See `xarray.testing.assert_allclose` for details).

    """
    _assert(
        isinstance(left, ndcollection.NDResultCollection),
        "left is not an NDResultCollection",
    )

    if left is right:
        return

    _assert(
        isinstance(right, ndcollection.NDResultCollection),
        "right is not an NDResultCollection",
    )

    _assert(len(left) == len(right), "length mismatch")

    for idx, (left_ndres, right_ndres) in enumerate(zip(left, right)):
        try:
            assert_ndresult_allclose(
                left_ndres,
                right_ndres,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                decode_bytes=decode_bytes,
            )
        except AssertionError as e:
            raise AssertionError(
                f"NDResultCollection[{idx}] mismatch: {e.args[0]}"
            )
