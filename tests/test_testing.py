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

"""Tests for skneuromsi/testing.py"""

# =============================================================================
# IMPORTS
# =============================================================================


import pytest

from skneuromsi import testing


# =============================================================================
# TESTS
# =============================================================================


def test_assert_ndresult_same_object(random_ndresult):
    ndres = random_ndresult()
    testing.assert_ndresult_allclose(ndres, ndres)


def test_assert_ndresult_collection_same_object(random_ndresultcollection):
    collection = random_ndresultcollection(length_max=2, length_min=2)
    testing.assert_ndresult_collection_allclose(collection, collection)


def test_assert_ndresult_allclose_equals(random_ndresult):
    ndres0 = random_ndresult(seed=42)
    ndres1 = random_ndresult(seed=42)
    assert ndres0 is not ndres1
    testing.assert_ndresult_allclose(ndres0, ndres1)


def test_assert_ndresult_collection_allclose_equals(random_ndresultcollection):
    collection0 = random_ndresultcollection(
        seed=42, length_max=2, length_min=2
    )
    collection1 = random_ndresultcollection(
        seed=42, length_max=2, length_min=2
    )
    assert collection0 is not collection1
    testing.assert_ndresult_collection_allclose(collection0, collection1)


def test_assert_ndresult_allclose_equals_different_seed(random_ndresult):
    ndres0 = random_ndresult(seed=42)
    ndres1 = random_ndresult(seed=43)
    with pytest.raises(AssertionError):
        testing.assert_ndresult_allclose(ndres0, ndres1)


def test_assert_ndresult_collection_allclose_different_seed(
    random_ndresultcollection,
):
    collection0 = random_ndresultcollection(
        seed=42, length_max=2, length_min=2
    )
    collection1 = random_ndresultcollection(
        seed=43, length_max=2, length_min=2
    )
    with pytest.raises(AssertionError):
        testing.assert_ndresult_collection_allclose(collection0, collection1)
