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

"""pytest configuration file."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools

import numpy as np

import pytest

from skneuromsi.core.ndresult import modes_to_data_array, NDResult

# =============================================================================
# FIXTURES
# =============================================================================


def check_min_max(min_name, max_name, minv, maxv, min_limit, max_limit):
    min_limit = minv if min_limit is None else min_limit
    max_limit = maxv if max_limit is None else max_limit

    if minv < min_limit:
        raise ValueError(f"{min_name!r} must be >= {min_limit!r}")
    if maxv > max_limit:
        raise ValueError(f"{max_name!r} must be <= {max_limit!r}")
    if minv > maxv:
        raise ValueError(f"{min_name!r} must be <= {max_name!r}")


def get_input_modes(random, input_modes_min, input_modes_max):
    check_min_max(
        "input_modes_min",
        "input_modes_max",
        input_modes_min,
        input_modes_max,
        1,
        None,
    )
    number = random.integers(input_modes_min, input_modes_max, endpoint=True)
    return tuple(f"Mode_{i}" for i in range(number))


def get_times_number(random, times_min, times_max):
    check_min_max("times_min", "times_max", times_min, times_max, 1, None)
    number = random.integers(times_min, times_max, endpoint=True)
    return number


def get_positions_number(random, positions_min, positions_max):
    check_min_max(
        "positions_min", "positions_max", positions_min, positions_max, 1, None
    )
    number = random.integers(positions_min, positions_max, endpoint=True)
    return number


def get_position_coordinates_number(
    random, position_coordinates_min, position_coordinates_max
):
    check_min_max(
        "position_coordinates_min",
        "position_coordinates_max",
        position_coordinates_min,
        position_coordinates_max,
        1,
        None,
    )
    number = random.integers(
        position_coordinates_min, position_coordinates_max, endpoint=True
    )
    return number


def make_mode_values(random, *, times, positions, position_coordinates, dtype):
    mode = []
    for i in range(position_coordinates):
        mode.append(random.random((times, positions), dtype=dtype))
    return mode[0] if position_coordinates == 1 else tuple(mode)


def make_modes_dict(
    random, *, input_modes, times, positions, position_coordinates, dtype
):
    modes = {}
    for mode in input_modes:
        modes[mode] = make_mode_values(
            random,
            times=times,
            positions=positions,
            position_coordinates=position_coordinates,
            dtype=dtype,
        )

    modes["output"] = make_mode_values(
        random,
        times=times,
        positions=positions,
        position_coordinates=position_coordinates,
        dtype=dtype,
    )

    return modes


@pytest.fixture(scope="session")
def random_modes_dict():

    def maker(
        *,
        dtype=np.float32,
        input_modes_min=1,
        input_modes_max=3,
        times_min=1,
        times_max=2000,
        positions_min=10,
        positions_max=50,
        position_coordinates_min=1,
        position_coordinates_max=3,
        seed=None,
    ):
        random = np.random.default_rng(seed)

        # how many keys in the dictionary
        input_modes = get_input_modes(random, input_modes_min, input_modes_max)

        # how many rows in the mode
        times = get_times_number(random, times_min, times_max)

        # how many columns in the mode
        positions = get_positions_number(random, positions_min, positions_max)

        # how many matrices for each mode
        position_coordinates = get_position_coordinates_number(
            random, position_coordinates_min, position_coordinates_max
        )

        # generate the dictionary
        modes_dict = make_modes_dict(
            random,
            input_modes=input_modes,
            times=times,
            positions=positions,
            position_coordinates=position_coordinates,
            dtype=dtype,
        )

        return modes_dict

    return maker


@pytest.fixture(scope="session")
def random_modes_da(random_modes_dict):
    def maker(*, dtype=np.float32, seed=None, **kwargs):
        random = np.random.default_rng(seed)
        modes_dict = random_modes_dict(seed=random, **kwargs)
        return modes_to_data_array(modes_dict, dtype=dtype)

    return maker


@pytest.fixture(scope="session")
def random_ndresult(random_modes_da):

    idx = 0

    def maker(
        *,
        mname=None,
        mtype="test",
        nmap=None,
        time_range=(0, 1000),
        position_range=(0, 20),
        time_res=0.1,
        position_res=1,
        run_params=None,
        extra=None,
        causes=None,
        seed=None,
        **kwargs,
    ):

        random = np.random.default_rng(seed)

        if mname is None:
            nonlocal idx
            mname = f"Model{idx}"
            idx += 1

        mtype = mtype
        output_mode = "output"
        nmap = {} if extra is None else dict(nmap)
        time_res = float(time_res)
        position_res = float(position_res)
        run_params = {} if run_params is None else dict(run_params)
        extra = {} if extra is None else dict(extra)

        times = np.max(time_range) * time_res
        positions = np.max(position_range) * position_res

        nddata = random_modes_da(
            seed=random,
            times_min=times,
            times_max=times,
            positions_min=positions,
            positions_max=positions,
            **kwargs,
        )

        causes = (
            random.integers(0, len(nddata.modes) - 1, endpoint=True)
            if causes is None
            else causes
        )

        return NDResult(
            mname=mname,
            mtype=mtype,
            output_mode=output_mode,
            nmap=nmap,
            nddata=nddata,
            time_range=time_range,
            position_range=position_range,
            time_res=time_res,
            position_res=position_res,
            causes=causes,
            run_params=run_params,
            extra=extra,
            ensure_dtype=None,
        )

    return maker
