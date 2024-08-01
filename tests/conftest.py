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

import xarray as xr

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def random_dataarray():
    def maker(dims, *, dtype=float, coords=None, attrs=None, seed=None):
        """Generate a random xarray DataArray with specified dimensions and \
        data type.

        Parameters
        ----------
        dims : dict
            A dictionary specifying the dimensions of the dataarray.
            The keys are the dimension names, and the values are the sizes.
        dtype : numpy.dtype, optional
            The data type of the random values. Default is float.
        coords : dict, optional
            A dictionary specifying the coordinates of the dataarray.
            The keys are the coordinate names, and the values are 1D arrays
            or dictionaries specifying the coordinate values.
        attrs : dict, optional
            A dictionary specifying the attributes of the dataarray.
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        xarray.DataArray
            A randomly generated xarray DataArray with the specified
            dimensions, data type, coordinates, and attributes.

        """
        random = np.random.default_rng(seed)

        shape = tuple(dims.values())
        data = random.random(size=shape, dtype=dtype)

        if coords is None:
            coords = {dim: np.arange(size) for dim, size in dims.items()}
        else:
            for coord_name, coord_vals in coords.items():
                if isinstance(coord_vals, dict):
                    coord_dims = coord_vals.get("dims", (coord_name,))
                    coord_data = coord_vals.get(
                        "data", random.random(size=dims[coord_name])
                    )
                    coords[coord_name] = xr.DataArray(
                        coord_data, dims=coord_dims
                    )

        return xr.DataArray(
            data, dims=list(dims.keys()), coords=coords, attrs=attrs
        )

    return maker


@pytest.fixture(scope="session")
def random_ndresult():

    model_idx = -1

    def make(
        mname=None,
        mtype=None,
        nmap=None,
        min_input_modes = 2,
        max_input_modes = 3,
        time_range=None,
        position_range=None,
        time_res=None,
        position_res=None,
        causes=None,
        run_params=None,
        extra=None,
        ensure_dtype=None,
        seed=None,
    ):

        if min_input_modes < 1:
            raise ValueError("min_input_mode must be >= 1")

        model_idx += 1
        random = np.random.default_rng(seed)

        mname = f"TestModel{model_idx}" if mname is None else mname



        # modes
        modes = ["mode_{}" mode_idx for mode in range(modes)]
        output_mode ["output_mode"]




    return make
