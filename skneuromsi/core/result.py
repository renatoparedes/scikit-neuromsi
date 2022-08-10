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
from collections.abc import Mapping
from typing import Iterable

import numpy as np

import pandas as pd

import xarray as xr

from .constants import (
    D_MODES,
    D_TIMES,
    D_POSITIONS,
    D_POSITIONS_COORDINATES,
    DIMENSIONS,
    XA_NAME,
)
from .plot import ResultPlotter
from .stats import ResultStatsAccessor

# =============================================================================
# CLASS EXTRA
# =============================================================================


class _Extra(Mapping):
    def __init__(self, d):
        self._data = dict(d)

    def __getitem__(self, d):
        return self._data[d]

    def __getattr__(self, d):
        return self._data[d]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        content = ", ".join(self._data.keys())
        return f"extra({content})"

    def __dir__(self):
        return super().__dir__() + list(self._data.keys())


# =============================================================================
# CLASS RESULT
# =============================================================================


class NDResult:
    def __init__(
        self, *, mname, mtype, nmap, nddata, time_res, position_res, extra
    ):
        self._mname = mname
        self._mtype = mtype
        self._nmap = dict(nmap)
        self._nddata = (
            modes_to_xarray(nddata) if isinstance(nddata, dict) else nddata
        )
        self._time_res = time_res
        self._position_res = position_res
        self._extra = _Extra(extra)

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
    def time_res(self):
        return self._time_res

    @property
    def position_res(self):
        return self._position_res

    @property
    def extra_(self):
        return self._extra

    e_ = extra_

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

    def _dim_as_dataframe(self, flt, dim_name):
        xa, dfs = self._nddata.sel({dim_name: flt}), []

        for gname, group in xa.groupby(dim_name):
            partial_df = group.to_dataframe(name=gname)
            if dim_name in partial_df.columns:
                partial_df = partial_df.drop(dim_name, axis="columns")
            else:
                partial_df = partial_df.droplevel(dim_name)

            dfs.append(partial_df)

        df = pd.concat(dfs, axis="columns")
        df.columns.name = dim_name
        return df

    def get_modes(self, include=None):
        flt = self._coherce_filters(include, self.modes_, D_MODES)
        df = self._dim_as_dataframe(flt, D_MODES)
        return df

    def get_times(self, include=None):
        flt = self._coherce_filters(include, self.times_, D_TIMES)
        df = self._dim_as_dataframe(flt, D_TIMES)
        return df

    def get_positions(self, include=None):
        flt = self._coherce_filters(include, self.positions_, D_POSITIONS)
        df = self._dim_as_dataframe(flt, D_POSITIONS)
        return df

    def get_positions_coordinates(self, include=None):
        flt = self._coherce_filters(
            include, self.positions_coordinates_, D_POSITIONS_COORDINATES
        )
        df = self._dim_as_dataframe(flt, D_POSITIONS_COORDINATES)
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
