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

"""pytest configuration file."""

# =============================================================================
# IMPORTS
# =============================================================================

import copy

import matplotlib

import numpy as np

import pytest

from skneuromsi.core.ndresult import (
    NDResult,
    modes_to_data_array,
)
from skneuromsi.ndcollection import NDResultCollection

import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

matplotlib.use("agg")

# =============================================================================
# FIXTURES
# =============================================================================


def check_min_max(min_name, max_name, minv, maxv, min_limit, max_limit):
    """
    Check if the given min and max values are within specified limits.

    Parameters
    ----------
    min_name : str
        Name of the minimum value parameter.
    max_name : str
        Name of the maximum value parameter.
    minv : int or float
        Minimum value to check.
    maxv : int or float
        Maximum value to check.
    min_limit : int or float or None
        Lower limit for the minimum value.
    max_limit : int or float or None
        Upper limit for the maximum value.

    Raises
    ------
    ValueError
        If any of the checks fail.
    """
    min_limit = minv if min_limit is None else min_limit
    max_limit = maxv if max_limit is None else max_limit

    if minv < min_limit:
        raise ValueError(f"{min_name!r} must be >= {min_limit!r}")
    if maxv > max_limit:
        raise ValueError(f"{max_name!r} must be <= {max_limit!r}")
    if minv > maxv:
        raise ValueError(f"{min_name!r} must be <= {max_name!r}")


def get_input_modes(random, input_modes_min, input_modes_max):
    """
    Generate a tuple of input mode names.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    input_modes_min : int
        Minimum number of input modes.
    input_modes_max : int
        Maximum number of input modes.

    Returns
    -------
    tuple of str
        Tuple of generated input mode names.
    """
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
    """
    Generate a random number of time points.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    times_min : int
        Minimum number of time points.
    times_max : int
        Maximum number of time points.

    Returns
    -------
    int
        Random number of time points.
    """
    check_min_max("times_min", "times_max", times_min, times_max, 1, None)
    number = random.integers(times_min, times_max, endpoint=True)
    return number


def get_positions_number(random, positions_min, positions_max):
    """
    Generate a random number of positions.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    positions_min : int
        Minimum number of positions.
    positions_max : int
        Maximum number of positions.

    Returns
    -------
    int
        Random number of positions.
    """
    check_min_max(
        "positions_min", "positions_max", positions_min, positions_max, 1, None
    )
    number = random.integers(positions_min, positions_max, endpoint=True)
    return number


def get_position_coordinates_number(
    random, position_coordinates_min, position_coordinates_max
):
    """
    Generate a random number of position coordinates.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    position_coordinates_min : int
        Minimum number of position coordinates.
    position_coordinates_max : int
        Maximum number of position coordinates.

    Returns
    -------
    int
        Random number of position coordinates.
    """
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
    """
    Generate random mode values.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    times : int
        Number of time points.
    positions : int
        Number of positions.
    position_coordinates : int
        Number of position coordinates.
    dtype : numpy.dtype
        Data type for the generated values.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Generated mode values.
    """
    mode = []
    for i in range(position_coordinates):
        mode.append(random.random((times, positions), dtype=dtype))
    return mode[0] if position_coordinates == 1 else tuple(mode)


def make_modes_dict(
    random, *, input_modes, times, positions, position_coordinates, dtype
):
    """
    Generate a dictionary of mode values.

    Parameters
    ----------
    random : numpy.random.Generator
        Random number generator.
    input_modes : tuple of str
        Input mode names.
    times : int
        Number of time points.
    positions : int
        Number of positions.
    position_coordinates : int
        Number of position coordinates.
    dtype : numpy.dtype
        Data type for the generated values.

    Returns
    -------
    dict
        Dictionary of generated mode values.
    """
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
    """Fixture for generating random mode dictionaries."""

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
        """
        Generate random mode dictionaries.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type for the generated values.
        input_modes_min : int
            Minimum number of input modes.
        input_modes_max : int
            Maximum number of input modes.
        times_min : int
            Minimum number of time points.
        times_max : int
            Maximum number of time points.
        positions_min : int
            Minimum number of positions.
        positions_max : int
            Maximum number of positions.
        position_coordinates_min : int
            Minimum number of position coordinates.
        position_coordinates_max : int
            Maximum number of position coordinates.
        seed : int
            Seed for the random number generator. Defaults to None.

        Returns
        -------
        dict
            Dictionary of generated mode values.

        """
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
    """Fixture for generating random mode DataArrays."""

    def maker(*, dtype=np.float32, seed=None, **kwargs):
        """Generate random "mode" DataArrays.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type for the generated values.
        seed : int
            Seed for the random number generator.
        kwargs
            Keyword arguments for random_modes_dict.

        Returns
        -------
        xarray.DataArray
            Random mode DataArray.

        """
        modes_dict = random_modes_dict(seed=seed, dtype=dtype, **kwargs)
        return modes_to_data_array(modes_dict, dtype=dtype)

    return maker


@pytest.fixture(scope="session")
def random_ndresult(random_modes_da):
    """Fixture for generating random NDResult objects."""

    def maker(
        *,
        mname=None,
        mtype="test",
        nmap=None,
        time_range=(0, 10),
        position_range=(0, 20),
        time_res=0.1,
        position_res=1,
        run_parameters=None,
        extra=None,
        causes=None,
        seed=None,
        **kwargs,
    ):
        """Generate a random `NDResult` object.

        Parameters
        ----------
        mname : str, optional
            Model name. If not provided, a default name will be generated.
        mtype : str, default 'test'
            Model type.
        nmap : dict, optional
            Dictionary mapping names to modes. If not provided, an empty
            dictionary will be used.
        time_range : tuple, optional
            Range of times for the model. Defaults to (0, 10).
        position_range : tuple, optional
            Range of positions for the model. Defaults to (0, 20).
        time_res : float, default 0.1
            Time resolution of the model.
        position_res : float, default 1
            Position resolution of the model.
        run_parameters : dict, optional
            Dictionary of run parameters. If not provided, an empty dictionary
            will be used.
        extra : dict, optional
            Dictionary of extra parameters. If not provided, an empty
            dictionary will be used.
        causes : list, optional
            List of causes. If not provided, a random list of causes will be
            generated.
        seed : int, optional
            Seed for the random number generator. If not provided, a
            random seed will be used.
        **kwargs :
            Additional keyword arguments to `random_modes_da`.

        Returns
        -------
        NDResult
            Random `NDResult` object with the specified parameters.

        """
        random = np.random.default_rng(seed)

        if mname is None:
            iinfo = np.iinfo(int)
            idx = random.integers(0, iinfo.max, dtype=int, endpoint=True)
            mname = f"Model{idx}"

        mtype = mtype
        output_mode = "output"
        nmap = {} if nmap is None else dict(nmap)
        time_res = float(time_res)
        position_res = float(position_res)
        run_parameters = {} if run_parameters is None else dict(run_parameters)
        extra = {} if extra is None else dict(extra)

        times = int(np.abs(np.subtract(*time_range)) / time_res)
        positions = int(np.abs(np.subtract(*position_range)) / position_res)

        nddata = random_modes_da(
            seed=random,
            times_min=times,
            times_max=times,
            positions_min=positions,
            positions_max=positions,
            **kwargs,
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
            run_parameters=run_parameters,
            extra=extra,
            ensure_dtype=None,
        )

    return maker


@pytest.fixture(scope="session")
def random_ndcollection(random_ndresult, silenced_tqdm_cls):
    def maker(
        *,
        size=2,
        input_modes=2,
        position_coordinates=3,
        run_parameters=None,
        sweep_parameter=None,
        causes=None,
        seed=None,
        **kwargs,
    ):
        random = np.random.default_rng(seed=seed)
        causes = (
            np.asarray(causes)
            if isinstance(causes, (list, tuple, np.ndarray))
            else np.array([causes])
        )
        run_parameters = {} if run_parameters is None else run_parameters

        kwargs.update(
            input_modes_min=input_modes,
            input_modes_max=input_modes,
            position_coordinates_min=position_coordinates,
            position_coordinates_max=position_coordinates,
            seed=random,
        )

        def generator():
            for sweep in range(size):
                curr_causes = random.choice(causes)
                curr_run_parameters = copy.deepcopy(run_parameters)
                if sweep_parameter is not None:
                    curr_run_parameters[sweep_parameter] = sweep
                coll = random_ndresult(
                    causes=curr_causes,
                    run_parameters=curr_run_parameters,
                    **kwargs,
                )
                yield coll

        return NDResultCollection.from_ndresults(
            "collection", generator(), tqdm_cls=silenced_tqdm_cls
        )

    return maker


# =============================================================================
# TQDM
# =============================================================================


class _SilencedTQDM(tqdm.tqdm):
    """Silence TQDM.

    This class is used to silence TQDM progress bars.

    """

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


@pytest.fixture(scope="session")
def silenced_tqdm_cls():
    return _SilencedTQDM
