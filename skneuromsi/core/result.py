#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import numbers
from turtle import back
from typing import Iterable

import numpy as np

import pandas as pd

import xarray as xr

from .plot import ResultPlotter
from .stats import ResultStatsAccessor

# =============================================================================
# CONSTANTS
# =============================================================================

D_MODES = "modes"
D_TIMES = "times"
D_POSITIONS = "positions"
D_POSITIONS_COORDINATES = "positions_coordinates"

DIMENSIONS = np.array([D_MODES, D_TIMES, D_POSITIONS, D_POSITIONS_COORDINATES])

XA_NAME = "values"

# =============================================================================
# CLASS RESULT
# =============================================================================


class NDResult:
    def __init__(self, *, mname, mtype, nmap, nddata):
        self._mname = mname
        self._mtype = mtype
        self._nmap = dict(nmap)
        self._nddata = (
            modes_to_xarray(nddata) if isinstance(nddata, dict) else nddata
        )

    # PROPERTIES ==============================================================

    @property
    def mname(self):
        return self._mname

    @property
    def mtype(self):
        return self._mtype

    @property
    def dims(self):
        return DIMENSIONS.copy()

    @property
    def nmap_(self):
        return self._nmap.copy()

    @property
    def modes_(self):
        return self._nddata[D_MODES].to_numpy()

    @property
    def times_(self):
        return self._nddata[D_TIMES].to_numpy()

    @property
    def positions_(self):
        return self._nddata[D_POSITIONS].to_numpy()

    @property
    def positions_coordinates_(self):
        return self._nddata[D_POSITIONS_COORDINATES].to_numpy()

    pcoords_ = positions_coordinates_

    # UTILS ===================================================================

    def __repr__(self):

        cls_name = type(self).__name__
        mname = self.mname
        modes = self.modes_
        _, times, pos, pos_coords = self._nddata.shape

        return (
            f"{cls_name}({mname}, modes={modes!s}, "
            f"times={times}, positions={pos}, "
            f"positions_coordinates={pos_coords})"
        )

    def to_xarray(self):
        return self._nddata.copy()

    def to_frame(self):
        return self._nddata.to_dataframe()

    # ACCESSORS ===============================================================

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        """Plot accessor."""
        return ResultPlotter(self)

    @property
    @functools.lru_cache(maxsize=None)
    def stats(self):
        """Stats accessor."""
        return ResultStatsAccessor(self)

    # DF BY DIMENSION =========================================================
    def _coherce_filters(self, flt, defaults, dim_name):
        if flt is None:
            return list(defaults)

        if isinstance(flt, (str, int, float, np.number)):
            flt = [flt]
        elif isinstance(flt, Iterable):
            flt = list(flt)

        diff = set(flt).difference(defaults)
        if diff:
            raise ValueError(f"{dim_name}{diff!r} not found")

        return flt

    def _dim_as_dataframe(self, flt, dim_name, rename_values):
        xa, dfs = self._nddata.sel({dim_name: flt}), []

        for gname, group in xa.groupby(dim_name):
            name = gname if rename_values else None
            partial_df = group.to_dataframe(name=name)
            if dim_name in partial_df.columns:
                partial_df = partial_df.drop(dim_name, axis="columns")
            else:
                partial_df = partial_df.droplevel(dim_name)

            dfs.append(partial_df)

        df = pd.concat(dfs, axis="columns")
        df.columns.name = dim_name
        return df

    def get_modes(self, include=None, *, rename_values=True):
        flt = self._coherce_filters(include, self.modes_, D_MODES)
        df = self._dim_as_dataframe(flt, D_MODES, rename_values)
        return df

    def get_times(self, include=None, *, rename_values=True):
        flt = self._coherce_filters(include, self.times_, D_TIMES)
        df = self._dim_as_dataframe(flt, D_TIMES, rename_values)
        return df

    def get_positions(self, include=None, *, rename_values=True):
        flt = self._coherce_filters(include, self.positions_, D_POSITIONS)
        df = self._dim_as_dataframe(flt, D_POSITIONS, rename_values)
        return df

    def get_positions_coordinates(self, include=None, *, rename_values=True):
        flt = self._coherce_filters(
            include, self.positions_coordinates_, D_POSITIONS_COORDINATES
        )
        df = self._dim_as_dataframe(
            flt, D_POSITIONS_COORDINATES, rename_values
        )
        return df

    get_pcoords = get_positions_coordinates


# =============================================================================
# UTILITIES
# =============================================================================


def modes_to_xarray(nddata):
    modes, coords = [], None

    # we iterate over each mode
    for mode_name, mode_coords in nddata.items():

        # NDResult always expects to have more than one coordinate per
        # position. If it has only one coordinate, it puts it into a
        # collection of length 1, so that it can continue te operations.
        if not isinstance(mode_coords, tuple):
            mode_coords = (mode_coords,)

        # we merge all the matrix of modes in a single 3D array
        # for example if we have two coordinates
        # x0 = [[1, 2, 3],
        #       [4, 5, 6]]
        # x1 = [[10, 20, 30],
        #       [40, 50, 60]]
        # np.dstack((x0, x1))
        # [[[1, 10], [2, 20], [3, 30]],
        #  [[4, 40], [5, 50], [6, 60]]]
        nd_mode_coords = np.dstack(mode_coords)

        if coords is None:  # first time we need to populate the indexes

            # retrieve how many times, positions and
            # position coordinates has the modes
            times_n, positions_n, pcoords_n = np.shape(nd_mode_coords)

            # we create the indexes for each dimension
            coords = [
                [],  # modes
                np.arange(times_n),  # times
                np.arange(positions_n),  # positions
                [f"x{idx}" for idx in range(pcoords_n)],  # pcoords
            ]

        # we add the mode name to the mode indexes
        coords[0].append(mode_name)

        # here we add the mode as the first dimension
        final_shape = (1,) + nd_mode_coords.shape

        # here we add the
        modes.append(nd_mode_coords.reshape(final_shape))

    xa = xr.DataArray(
        np.concatenate(modes),
        coords=coords,
        dims=DIMENSIONS,
        name=XA_NAME,
    )

    return xa
