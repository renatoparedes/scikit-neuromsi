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
