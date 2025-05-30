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

"""Tests for the `skneuromsi.io` module."""

# =============================================================================
# IMPORTS
# =============================================================================

import io
import json
import zipfile

import pytest

import skneuromsi as sknmsi

# =============================================================================
# TESTS
# =============================================================================


def test_NDResult_store_and_open(random_ndresult):
    ndres = random_ndresult()

    buffer = io.BytesIO()

    sknmsi.store_ndresult(buffer, ndres)
    buffer.seek(0)
    restored = sknmsi.read_ndr(buffer)

    sknmsi.testing.assert_ndresult_allclose(ndres, restored)


def test_NDResultCollection_store_and_open(random_ndresult, silenced_tqdm_cls):
    ndres = random_ndresult()
    collection = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres, ndres], tqdm_cls=silenced_tqdm_cls
    )

    buffer = io.BytesIO()

    sknmsi.to_ndc(buffer, collection, tqdm_cls=silenced_tqdm_cls)
    buffer.seek(0)
    restored = sknmsi.read_ndc(buffer, tqdm_cls=silenced_tqdm_cls)

    sknmsi.testing.assert_ndresult_collection_allclose(collection, restored)


def test_store_ndresults_collection_not_instace_of_ndcollection():
    with pytest.raises(TypeError):
        sknmsi.to_ndc("", None)


def test_store_ndresults_not_instace_of_ndresult():
    with pytest.raises(TypeError):
        sknmsi.to_ndr("", None)


def test_open_ndresults_invalid_object_type():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_fp:
        fake_metadata = json.dumps({"object_type": "fake"})
        zip_fp.writestr("metadata.json", fake_metadata)
    buffer.seek(0)

    with pytest.raises(ValueError):
        sknmsi.read_ndc(buffer)


def test_open_ndresults_invalid_object_size():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_fp:
        fake_metadata = json.dumps(
            {
                "object_type": "ndcollection",
                "object_size": 3,
                "object_kwargs": {},
            }
        )
        zip_fp.writestr("metadata.json", fake_metadata)
    buffer.seek(0)

    with pytest.raises(ValueError):
        sknmsi.read_ndc(buffer, expected_size=1)
