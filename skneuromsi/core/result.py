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

from typing import Iterable

import methodtools

import numpy as np

import pandas as pd

import xarray as xr

from .constants import (
    DIMENSIONS,
    D_MODES,
    D_POSITIONS,
    D_POSITIONS_COORDINATES,
    D_TIMES,
    XA_NAME,
)
from .plot import ResultPlotter
from .stats import ResultStatsAccessor
from ..utils import Bunch

# =============================================================================
# CLASS RESULT
# =============================================================================


class NDResult:
    def __init__(
        self,
        *,
        mname,
        mtype,
        output_mode,
        nmap,
        nddata,
        time_range,
        position_range,
        time_res,
        position_res,
        causes,
        run_params,
        extra,
    ):
        self._mname = mname
        self._mtype = mtype
        self._output_mode = output_mode
        self._nmap = dict(nmap)
        self._nddata = (
            modes_to_data_array(nddata) if isinstance(nddata, dict) else nddata
        )
        self._time_range = np.asarray(time_range)
        self._position_range = np.asarray(position_range)
        self._time_res = time_res
        self._position_res = position_res
        self._time_res = time_res
        self._position_res = position_res
        self._run_params = Bunch("run_params", run_params)
        self._extra = Bunch("extra", extra)
        self._causes = causes

    # PROPERTIES ==============================================================

    @property
    def mname(self):
        return self._mname

    @property
    def mtype(self):
        return self._mtype

    @property
    def output_mode(self):
        return self._output_mode

    @property
    def dims(self):
        return DIMENSIONS.copy()

    @property
    def nmap_(self):
        return self._nmap.copy()

    @property
    def time_range(self):
        return self._time_range

    @property
    def position_range(self):
        return self._position_range

    @property
    def time_res(self):
        return self._time_res

    @property
    def position_res(self):
        return self._position_res

    @property
    def run_params(self):
        return self._run_params

    rp = run_params

    @property
    def extra_(self):
        return self._extra

    e_ = extra_

    @property
    def causes_(self):
        return self._causes

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
        causes = bool(self.causes_)

        return (
            f"<{cls_name} '{mname}', modes={modes!s}, "
            f"times={times}, positions={pos}, "
            f"positions_coordinates={pos_coords}, causes={causes}>"
        )

    def to_xarray(self):
        return self._nddata.copy()

    # ACCESSORS ===============================================================

    @property
    def plot(self):
        """Plot accessor."""
        return ResultPlotter(self)

    @property
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

    # IO ======================================================================
    def to_dict(self):
        return {
            "mname": str(self.mname),
            "mtype": str(self.mtype),
            "output_mode": str(self.output_mode),
            "nmap": self.nmap_,
            "time_range": self.time_range,
            "position_range": self.position_range,
            "time_res": self.time_res,
            "position_res": self.position_res,
            "causes": self.causes_,
            "run_params": self.run_params.to_dict(),
            "extra": self.extra_.to_dict(),
            "nddata": self.to_xarray(),
        }

    def to_netcdf(self, path_or_stream, metadata=None, **kwargs):
        from .. import io

        io.ndresult_to_netcdf(
            path_or_stream, self, metadata=metadata, **kwargs
        )


# =============================================================================
# UTILITIES
# =============================================================================


def modes_to_data_array(nddata):
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

    data = (
        np.concatenate(modes) if modes else np.array([], ndmin=len(DIMENSIONS))
    )

    xa = xr.DataArray(
        data,
        coords=coords,
        dims=DIMENSIONS,
        name=XA_NAME,
    )

    return xa
