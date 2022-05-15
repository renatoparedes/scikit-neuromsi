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
# CLASS RESULT
# =============================================================================


I_MODES = "modes"
I_DIMS = "dimensions"
I_TIMES = "times"


class NDResult:
    def __init__(self, *, mname, mtype, nmap, nddata):
        self._nddata = self._modes_to_xarray(nddata)

    def _modes_to_xarray(self, nddata):
        modes_xa = []

        # iteramos sobre cada uno de los modos
        for mode_name, mode_coords in nddata.items():

            # si solo hay una matriz en el modo lo metemos en una
            # tupla de un elemento
            if not isinstance(mode_coords, (list, tuple)):
                mode_coords = (mode_coords,)

            # we merge all the matrix of modes in a single cube
            nd_mode_coords = np.dstack(mode_coords)

            # vemos cuantos tiempos, posisiones y coordenadas hay
            times_n, positions_n, position_coords_n = np.shape(nd_mode_coords)

            # creamos los indices para cada dimension
            modes_names = [mode_name]
            times = np.arange(times_n)
            positions = np.arange(positions_n)
            pos_coords = np.array(
                [f"x{idx}" for idx in range(position_coords_n)]
            )

            # here we add the mode as the first dimension
            final_shape = (1,) + nd_mode_coords.shape

            mode_xa = xr.DataArray(
                nd_mode_coords.reshape(final_shape),
                coords=[modes_names, times, positions, pos_coords],
                dims=["mode", "time", "positions", "position_coordinates"],
                name=mode_name,
            )
            modes_xa.append(mode_xa)

        # mergeamos todos los modos en un unico DataArray
        xa = xr.combine_nested(modes_xa, "mode")

        return xa

    @property
    def mname(self):
        return self._mname

    @property
    def mtype(self):
        return self._mtype

    @property
    def nmap_(self):
        return self._nmap.copy()

    @property
    def modes_(self):
        return self._modes

    @property
    def dims_(self):
        return self._dims

    @property
    def ndims_(self):
        return self._ndims

    # @property
    # def times_(self):
    #     return np.array(self._df.index)

    # @property
    # def modes_(self):
    #     modes = set(self._df.columns.get_level_values(I_MODES))
    #     modes = sorted(modes)
    #     return np.array(modes)

    # @property
    # def positions_(self):
    #     positions = set(self._df.columns.get_level_values(I_POS))
    #     positions = sorted(positions)
    #     return np.array(positions)

    def __repr__(self):

        cls_name = type(self).__name__
        mname = self.mname
        modes = self.modes_
        times, pos = 1, 2

        return (
            f"{cls_name}({mname}, modes={modes}, "
            f"times={times}, positions={pos})"
        )

    def to_frame(self):
        return self._df.copy()

    def get_time(self, time):
        time_serie = self._df.loc[time].unstack()

        series = (time_serie.loc[mode] for mode in self.modes_)
        df = pd.ndDataFrame(series, index=self.modes_)
        df.index.name = I_MODES

        return df.T

    def get_mode(self, mode):
        return self._df.xs(mode, level=I_MODES, axis="columns")

    def get_position(self, position):
        return self._df.xs(position, level=I_POS, axis="columns")
