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
from turtle import back

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

    # DF BY DIMENSION =========================================================

    def get_mode(self, mode, *, rename_values=True):
        if mode not in self.modes_:
            raise ValueError(f"Mode {mode} not found")

        name = mode if rename_values else None

        df = self._nddata.sel({D_MODES: mode}).to_dataframe(name=name)
        del df[D_MODES]
        return df

    def get_time(self, time, *, rename_values=True):
        if time not in self.times_:
            raise ValueError(f"Time {time} not found")

        name = f"Time {time}" if rename_values else None

        df = self._nddata.sel({D_TIMES: time}).to_dataframe(name=name)
        del df[D_TIMES]
        return df

    def get_position(self, position, *, rename_values=True):
        if position not in self.positions_:
            raise ValueError(f"Position {position} not found")

        name = f"Position {position}" if rename_values else None

        df = self._nddata.sel({D_POSITIONS: position}).to_dataframe(name=name)
        del df[D_POSITIONS]
        return df

    def get_position_coordinate(self, coordinate, *, rename_values=True):
        if coordinate not in self.positions_coordinates_:
            raise ValueError(f"Position coordinate {coordinate} not found")

        name = coordinate if rename_values else None

        df = self._nddata.sel(
            {D_POSITIONS_COORDINATES: coordinate}
        ).to_dataframe(name=name)
        del df[D_POSITIONS_COORDINATES]
        return df

    get_pcoord = get_position_coordinate


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
