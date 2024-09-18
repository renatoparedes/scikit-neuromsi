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

"""test for skneuromsi.ndcollection.collection

"""

# =============================================================================
# IMPORTS
# =============================================================================

import io
from unittest import mock

import numpy as np

import pandas as pd

import pytest

import skneuromsi as sknmsi
from skneuromsi.ndcollection import bias_acc, causes_acc, collection, cplot_acc

# =============================================================================
# TESTS
# =============================================================================


def test_NDCollection(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p0": 1},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p0": 2},
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
    )

    assert coll.name == "collection"
    assert coll.tqdm_cls == silenced_tqdm_cls
    assert len(coll) == 2
    sknmsi.testing.assert_ndresult_allclose(coll[0], ndres0)
    sknmsi.testing.assert_ndresult_allclose(coll[1], ndres1)
    np.testing.assert_array_equal(coll.modes_, ["Mode_0", "output"])
    assert coll.output_mode_ == "output"
    assert coll.run_parameters_ == ("p0",)
    np.testing.assert_array_equal(coll.causes_, [1, 1])
    np.testing.assert_array_equal(
        coll.run_parameters_values, [{"p0": 1}, {"p0": 2}]
    )
    np.testing.assert_array_equal(coll.input_modes_, ["Mode_0"])
    assert repr(coll) == "<NDResultCollection 'collection' len=2>"


def test_NDCollection_empty_collection():
    expected_error = "Empty NDResultCollection not allowed"
    with pytest.raises(ValueError, match=expected_error):
        collection.NDResultCollection("collection", [])


def test_NDCollection_invalid_tqdm_class(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p0": 1},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p0": 2},
    )
    expected_error = "'tqdm_cls' must be an instance of tqdm.tqdm or None"
    with pytest.raises(TypeError, match=expected_error):
        collection.NDResultCollection.from_ndresults(
            "collection", [ndres0, ndres1], tqdm_cls=object
        )


def test_NDCollection_not_compressed_NDResult():
    expected_error = "Not all results are CompressedNDResult objects"
    with pytest.raises(ValueError, match=expected_error):
        collection.NDResultCollection("collection", [0, 0])


def test_NDCollection_invalid_common_metadata(
    random_ndresult, silenced_tqdm_cls
):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p0": 1},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        causes=1,
        run_parameters={"p1": 1},
    )
    expected_error = (
        r"All NDResults must have the same metadata in "
        r"\['dims', 'modes', 'output_mode', 'run_parameters'\]."
    )

    with pytest.raises(ValueError, match=expected_error):
        collection.NDResultCollection.from_ndresults(
            "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
        )


def test_NDCollection_asarray(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
    )

    asarray = coll[:]
    assert isinstance(asarray, np.ndarray)
    assert asarray.dtype == object
    sknmsi.testing.assert_ndresult_allclose(asarray[0], ndres0)
    sknmsi.testing.assert_ndresult_allclose(asarray[1], ndres1)


def test_NDCollection_disparity_matrix(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
    )

    expected = pd.DataFrame(
        [[0, "foo"], [1, "foo"]],
        columns=pd.Index(["p0", "p1"], name="Parameters"),
    )

    pd.testing.assert_frame_equal(coll.disparity_matrix(), expected)


def test_NDCollection_changing_parameter(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
    )
    ndres2 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1, ndres2], tqdm_cls=silenced_tqdm_cls
    )

    expected = pd.Series(
        [True, False],
        index=pd.Index(["p0", "p1"], name="Parameters"),
        name="Changes",
    )

    pd.testing.assert_series_equal(coll.changing_parameters(), expected)


def test_NDCollection_coerce_parameter(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
    )
    ndres2 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "faa"},
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
    )

    assert coll.coerce_parameter() == "p0"
    assert coll.coerce_parameter("p1") == "p1"

    with pytest.raises(ValueError, match="Unknown run_parameter 'foo'"):
        coll.coerce_parameter("foo")

    # no candidates
    coll_with_ambiguity = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres0], tqdm_cls=silenced_tqdm_cls
    )
    expected_error = (
        "The value of 'run_parameter' is ambiguous since it has 0 candidates."
    )
    with pytest.raises(ValueError, match=expected_error):
        coll_with_ambiguity.coerce_parameter()

    # to much candidates
    coll_with_ambiguity = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres2], tqdm_cls=silenced_tqdm_cls
    )
    expected_error = (
        r"The value of 'run_parameter' "
        r"is ambiguous since it has 2 candidates. Candidates: \['p0' 'p1'\]"
    )
    with pytest.raises(ValueError, match=expected_error):
        coll_with_ambiguity.coerce_parameter()


def test_NDCollection_modes_variance_sum(random_ndresult, silenced_tqdm_cls):

    seed = np.random.default_rng(42)

    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
        seed=seed,
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
        seed=seed,
    )
    ndres2 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "faa"},
        seed=seed,
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1, ndres2], tqdm_cls=silenced_tqdm_cls
    )

    expected = pd.Series(
        [0.2547706067562103, 0.24663656949996948],
        index=pd.Index(["Mode_0", "output"], name="modes"),
        name="VarSum",
        dtype=np.float32,
    )

    pd.testing.assert_series_equal(coll.modes_variance_sum(), expected)


def test_NDCollection_coerce_mode(random_ndresult, silenced_tqdm_cls):

    seed = np.random.default_rng(42)

    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
        seed=seed,
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
        seed=seed,
    )
    ndres2 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "faa"},
        seed=seed,
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1, ndres2], tqdm_cls=silenced_tqdm_cls
    )

    assert coll.coerce_mode() == "Mode_0"
    assert coll.coerce_mode("output") == "output"

    with pytest.raises(ValueError, match="Unknown mode 'foo'"):
        coll.coerce_mode("foo")

    # ambiguous mode

    to_patch = (
        "skneuromsi.ndcollection."
        "collection.NDResultCollection.modes_variance_sum"
    )
    same_modes_variance_sum = pd.Series(
        [0.2, 0.2],
        index=pd.Index(["Mode_0", "output"], name="modes"),
        name="VarSum",
        dtype=np.float32,
    )

    expected_error = (
        "The value of 'mode' is ambiguous since it "
        r"has 2 candidates. Candidates: \['Mode_0' 'output'\]"
    )

    with mock.patch(to_patch, return_value=same_modes_variance_sum):
        with pytest.raises(ValueError, match=expected_error):
            coll.coerce_mode()


def test_NDCollection_coerce_dimension(random_ndresult, silenced_tqdm_cls):

    seed = np.random.default_rng(42)

    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 0, "p1": "foo"},
        seed=seed,
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "foo"},
        seed=seed,
    )
    ndres2 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        run_parameters={"p0": 1, "p1": "faa"},
        seed=seed,
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1, ndres2], tqdm_cls=silenced_tqdm_cls
    )

    assert coll.coerce_dimension() == sknmsi.core.constants.D_TIMES
    assert (
        coll.coerce_dimension(sknmsi.core.constants.D_MODES)
        == sknmsi.core.constants.D_MODES
    )
    assert (
        coll.coerce_dimension(sknmsi.core.constants.D_TIMES)
        == sknmsi.core.constants.D_TIMES
    )
    assert (
        coll.coerce_dimension(sknmsi.core.constants.D_POSITIONS)
        == sknmsi.core.constants.D_POSITIONS
    )
    assert (
        coll.coerce_dimension(sknmsi.core.constants.D_POSITIONS_COORDINATES)
        == sknmsi.core.constants.D_POSITIONS_COORDINATES
    )

    with pytest.raises(ValueError, match="Unknown dimension 'foo'"):
        coll.coerce_dimension("foo")


def test_NDCollection_accessors(random_ndresult, silenced_tqdm_cls):
    ndres0 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    ndres1 = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )

    coll = collection.NDResultCollection.from_ndresults(
        "collection", [ndres0, ndres1], tqdm_cls=silenced_tqdm_cls
    )

    assert isinstance(coll.causes, causes_acc.NDResultCollectionCausesAcc)
    assert isinstance(coll.bias, bias_acc.NDResultCollectionBiasAcc)
    assert isinstance(coll.plot, cplot_acc.NDResultCollectionPlotter)


def test_NDResultCollection_to_ndc(random_ndresult, silenced_tqdm_cls):
    ndres = random_ndresult()
    collection = sknmsi.NDResultCollection.from_ndresults(
        "collection", [ndres, ndres], tqdm_cls=silenced_tqdm_cls
    )

    oo_buffer = io.BytesIO()
    collection.to_ndc(oo_buffer, quiet=True)
    oo_buffer.seek(0)

    func_buffer = io.BytesIO()
    sknmsi.to_ndc(func_buffer, collection, tqdm_cls=silenced_tqdm_cls)
    func_buffer.seek(0)

    restored_from_oo = sknmsi.read_ndc(oo_buffer, tqdm_cls=silenced_tqdm_cls)
    restored_from_func = sknmsi.read_ndc(
        func_buffer, tqdm_cls=silenced_tqdm_cls
    )

    sknmsi.testing.assert_ndresult_collection_allclose(
        restored_from_func, restored_from_oo
    )
