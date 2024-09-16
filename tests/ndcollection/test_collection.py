import numpy as np

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
        causes=None,
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
    np.testing.assert_array_equal(coll.causes_, [1, None])


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
