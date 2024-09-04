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

"""Tests for the `skneuromsi.io` module.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import io
import zipfile

import pytest

import skneuromsi as sknmsi

from .conftest import SilencedTQDM

# =============================================================================
# TESTS
# =============================================================================


def test_ndresult_store_and_open(random_ndresult):
    ndres = random_ndresult()

    buffer = io.BytesIO()

    sknmsi.store_ndresult(buffer, ndres)
    buffer.seek(0)
    restored = sknmsi.read_ndr(buffer)

    sknmsi.testing.assert_ndresult_allclose(ndres, restored)


def test_ndcollection_store_and_open(random_ndresultcollection):
    collection = random_ndresultcollection(length_max=2, length_min=2)

    buffer = io.BytesIO()

    sknmsi.to_ndc(buffer, collection, tqdm_cls=SilencedTQDM)
    buffer.seek(0)
    restored = sknmsi.read_ndc(buffer, tqdm_cls=SilencedTQDM)

    sknmsi.testing.assert_ndresult_collection_allclose(collection, restored)


def test_ndcollection_oostore_equivalent(random_ndresultcollection):
    collection = random_ndresultcollection(length_max=2, length_min=2)
    oo_buffer = io.BytesIO()
    collection.to_ndc(oo_buffer, quiet=True)
    oo_buffer.seek(0)

    func_buffer = io.BytesIO()
    sknmsi.to_ndc(func_buffer, collection, tqdm_cls=SilencedTQDM)
    func_buffer.seek(0)

    restored_from_oo = sknmsi.read_ndc(oo_buffer, tqdm_cls=SilencedTQDM)
    restored_from_func = sknmsi.read_ndc(func_buffer, tqdm_cls=SilencedTQDM)

    sknmsi.testing.assert_ndresult_collection_allclose(
        restored_from_func, restored_from_oo
    )


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
