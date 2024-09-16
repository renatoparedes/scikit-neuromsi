import numpy as np

import pandas as pd

import pytest

import skneuromsi as sknmsi
from skneuromsi.ndcollection import collection


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
