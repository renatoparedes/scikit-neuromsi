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

import io

import skneuromsi as sknmsi


def test_ndresult_store_and_open(random_ndresult):
    ndres = random_ndresult()

    buffer = io.BytesIO()

    sknmsi.to_ndr(buffer, ndres)
    buffer.seek(0)
    restored = sknmsi.read_ndr(buffer)

    sknmsi.testing.assert_ndresult_allclose(ndres, restored)


def test_ndresult_oostore_equivalent(random_ndresult):
    ndres = random_ndresult()

    oo_buffer = io.BytesIO()
    ndres.to_ndr(oo_buffer)
    oo_buffer.seek(0)

    func_buffer = io.BytesIO()
    sknmsi.to_ndr(func_buffer, ndres)
    func_buffer.seek(0)

    restored_from_oo = sknmsi.read_ndr(oo_buffer)
    restored_from_func = sknmsi.read_ndr(func_buffer)

    sknmsi.testing.assert_ndresult_allclose(
        restored_from_func, restored_from_oo
    )


def test_ndcollection_store_and_open(random_ndresultcollection):
    collection = random_ndresultcollection(length_max=2, length_min=2)

    buffer = io.BytesIO()

    sknmsi.to_ndc(buffer, collection, tqdm_cls=None)
    buffer.seek(0)
    restored = sknmsi.read_ndc(buffer, tqdm_cls=None)

    sknmsi.testing.assert_ndresult_collection_allclose(collection, restored)


def test_ndcollection_oostore_equivalent(random_ndresultcollection):
    collection = random_ndresultcollection(length_max=2, length_min=2)
    oo_buffer = io.BytesIO()
    collection.to_ndc(oo_buffer, quiet=True)
    oo_buffer.seek(0)

    func_buffer = io.BytesIO()
    sknmsi.to_ndc(func_buffer, collection, tqdm_cls=None)
    func_buffer.seek(0)

    restored_from_oo = sknmsi.read_ndc(oo_buffer, tqdm_cls=None)
    restored_from_func = sknmsi.read_ndc(func_buffer, tqdm_cls=None)

    sknmsi.testing.assert_ndresult_collection_allclose(
        restored_from_func, restored_from_oo
    )
