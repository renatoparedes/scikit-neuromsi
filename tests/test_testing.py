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

import skneuromsi as sknmsi
from skneuromsi import testing


# =============================================================================
# TESTS
# =============================================================================


def test_assert_ndresult_same_object(random_ndresult):
    ndres = random_ndresult()
    testing.assert_ndresult_allclose(ndres, ndres)


def test_assert_ndresult_collection_same_object(
    random_ndresult, silenced_tqdm_cls
):
    ndres = random_ndresult()
    collection = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres, ndres], tqdm_cls=silenced_tqdm_cls
    )
    testing.assert_ndresult_collection_allclose(collection, collection)


def test_assert_ndresult_allclose_equals(random_ndresult):
    ndres0 = random_ndresult(seed=42)
    ndres1 = random_ndresult(seed=42)
    assert ndres0 is not ndres1
    testing.assert_ndresult_allclose(ndres0, ndres1)


def test_assert_ndresult_collection_allclose_equals(
    random_ndresult, silenced_tqdm_cls
):
    ndres = random_ndresult()
    collection0 = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres, ndres], tqdm_cls=silenced_tqdm_cls
    )
    collection1 = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres, ndres], tqdm_cls=silenced_tqdm_cls
    )
    assert collection0 is not collection1
    testing.assert_ndresult_collection_allclose(collection0, collection1)


def test_assert_ndresult_allclose_equals_different_seed(random_ndresult):
    ndres0 = random_ndresult(seed=42)
    ndres1 = random_ndresult(seed=43)
    with pytest.raises(AssertionError):
        testing.assert_ndresult_allclose(ndres0, ndres1)


def test_assert_ndresult_collection_allclose_different_seed(
    random_ndresult, silenced_tqdm_cls
):
    ndres0 = random_ndresult(seed=42)
    collection0 = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres0], tqdm_cls=silenced_tqdm_cls
    )

    ndres1 = random_ndresult(seed=43)
    collection1 = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres1, ndres1], tqdm_cls=silenced_tqdm_cls
    )
    with pytest.raises(AssertionError):
        testing.assert_ndresult_collection_allclose(collection0, collection1)
