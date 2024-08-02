def test_coso(random_ndresult, random_modes_da, random_modes_dict):
    modes_dict = random_modes_dict(
        input_modes_min=2,
        input_modes_max=2,
        times_min=10,
        times_max=10,
        positions_max=5,
        positions_min=5,
        position_coordinates_min=2,
        position_coordinates_max=2,
        dtype=float,
        seed=42,
    )

    modes_da = random_modes_da(
        position_coordinates_min=2,
        position_coordinates_max=2,
        input_modes_min=2,
        input_modes_max=2,
        positions_max=5,
        positions_min=5,
        times_min=10,
        times_max=10,
        dtype=float,
        seed=42,
    )

    ndres = random_ndresult()

    import ipdb

    ipdb.set_trace()
