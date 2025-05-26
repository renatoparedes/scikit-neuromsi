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

"""Utilities to represents a multisensory integration result as a \
multidimensional array."""


# =============================================================================
# IMPORTS
# =============================================================================

from typing import Iterable

import numpy as np

import pandas as pd

import xarray as xa

from .plot_acc import ResultPlotter
from .stats_acc import ResultStatsAccessor
from ..constants import (
    DIMENSIONS,
    D_MODES,
    D_POSITIONS,
    D_POSITIONS_COORDINATES,
    D_TIMES,
    XA_NAME,
)
from ...utils import Bunch, ddtype_tools


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def modes_to_data_array(modes_dict, dtype):
    """Convert a dictionary of modes to an xarray.DataArray.

    Parameters
    ----------
    modes_dict : dict
        A dictionary of modes and their corresponding coordinates.
    dtype : numpy.dtype, optional
        The data type of the resulting xarray.DataArray.

    Returns
    -------
    xarray.DataArray
        The modes as an xarray.DataArray.

    """
    # we start with an empty array
    modes, coords = [], None

    # we iterate over each mode
    for mode_name, mode_coords in modes_dict.items():
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
        # The astype is to ensure that the data type is consistent
        nd_mode_coords = np.dstack(mode_coords).astype(dtype, copy=False)

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
    da = xa.DataArray(data, coords=coords, dims=DIMENSIONS, name=XA_NAME)

    return da


# =============================================================================
# CLASS RESULT
# =============================================================================


class NDResult:
    """Represents a multisensory integration result.

    Parameters
    ----------
    mname : str
        The name of the model.
    mtype : str
        The type of the model.
    output_mode : str
        The output mode of the model.
    nmap : dict
        A dictionary mapping modes to their corresponding values.
    nddata : xarray.DataArray or dict
        The multidimensional data as an xarray.DataArray.
    time_range : tuple
        The range of time values.
    position_range : tuple
        The range of position values.
    time_res : float
        The resolution of time values.
    position_res : float
        The resolution of position values.
    causes : int, float or None
        The number of causes in the result.
    run_parameters : dict
        The parameters used for running the model.
    extra : dict
        Extra information associated with the result.
    ensure_dtypes : numpy.dtype, optional (default=infer)
        Force all data types to be assigned to this type.
        This only applies to parameters that accept the dtype message
        If None, the data types are inferred.

    """

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
        run_parameters,
        extra,
        ensure_dtype=None,
    ):

        self._mname = str(mname)
        self._mtype = str(mtype)
        self._output_mode = str(output_mode)
        self._nmap = dict(nmap)
        self._time_range = np.asarray(time_range, dtype=ensure_dtype)
        self._position_range = np.asarray(position_range, dtype=ensure_dtype)
        self._time_res = float(time_res)
        self._position_res = float(position_res)
        self._run_parameters = dict(run_parameters)
        self._extra = dict(extra)
        self._causes = causes
        self._nddata = nddata

        # Ensure that the instance variables are not dynamically added.
        if ensure_dtype is not None:
            self.__dict__ = ddtype_tools.deep_astype(
                vars(self), dtype=ensure_dtype
            )

        self._validate()

    def _validate(self):
        """Validate the result data."""
        # chek if the output mode is pressent
        output_mode = self._output_mode
        nddata = self._nddata
        if output_mode not in nddata.modes:
            raise ValueError(f"Output mode '{output_mode}' not found.")

        # check if there are at least two modes
        if len(nddata.modes) < 2:
            raise ValueError("At least two modes are required.")

        # check time range size and limits
        trange = tuple(self._time_range)
        if len(trange) != 2 or trange[0] > trange[1]:
            raise ValueError(
                f"The time_range must be (min, max). Got {trange}"
            )

        # check if the time range and resolution match the data
        tres = self._time_res
        expected_times = int(np.abs(np.subtract(*trange)) / tres) or 1
        times = len(nddata.times)
        if expected_times != times:
            raise ValueError(
                "The time_range and time_res do not match the data. "
                f"Expected {expected_times} times, got {times}"
            )

        # check position range size and limits
        prange = tuple(self._position_range)
        if len(prange) != 2 or prange[0] > prange[1]:
            raise ValueError(
                f"The position_range must be (min, max). Got {prange}"
            )

        # check if the position range and resolution match the data
        pres = self._position_res
        expected_positions = int(np.abs(np.subtract(*prange)) / pres) or 1
        positions = len(self._nddata.positions)
        if expected_positions != positions:
            raise ValueError(
                "The position_range and position_res do not match the data. "
                f"Expected {expected_positions} positions, got {positions}"
            )

        # check causes
        causes = self._causes
        if not (causes is None or isinstance(causes, (int, float, np.number))):
            raise ValueError(
                f"causes must be, int, float or None, got {type(causes)}"
            )

    @classmethod
    def from_modes_dict(cls, *, modes_dict, ensure_dtype=None, **kwargs):
        """Create an NDResult object from a dictionary of modes.

        Parameters
        ----------
        modes_dict : dict
            A dictionary mapping modes to their corresponding values.
        ensure_dtype : numpy.dtype, optional
            Force all data types to be assigned to this type.
            This only applies to parameters that accept the dtype message
            If None, the data types are inferred.
        **kwargs
            Additional keyword arguments to pass to the NDResult constructor.

        Returns
        -------
        NDResult
            The NDResult object.

        """
        nddata = modes_to_data_array(modes_dict, dtype=ensure_dtype)
        return cls(nddata=nddata, ensure_dtype=ensure_dtype, **kwargs)

    # PROPERTIES ==============================================================

    @property
    def mname(self):
        """str: The name of the model."""
        return self._mname

    @property
    def mtype(self):
        """str: The type of the model."""
        return self._mtype

    @property
    def output_mode(self):
        """str: The output mode of the model."""
        return self._output_mode

    @property
    def dims(self):
        """list: The dimensions of the result data."""
        return DIMENSIONS.copy()

    @property
    def nmap_(self):
        """dict: A copy of the nmap dictionary."""
        return self._nmap.copy()

    @property
    def time_range(self):
        """tuple: The range of time values."""
        return self._time_range

    @property
    def position_range(self):
        """tuple: The range of position values."""
        return self._position_range

    @property
    def time_res(self):
        """float: The resolution of time values."""
        return self._time_res

    @property
    def position_res(self):
        """float: The resolution of position values."""
        return self._position_res

    @property
    def run_parameters(self):
        """Bunch: The parameters used for running the model."""
        return Bunch("run_parameters", self._run_parameters)

    rp = run_parameters

    # dtypes are at the end <===================================!!!

    @property
    def extra_(self):
        """Bunch: Extra information associated with the result."""
        return Bunch("extra", self._extra)

    e_ = extra_

    @property
    def causes_(self):
        """int: The number of causes in the result."""
        return self._causes

    @property
    def modes_(self):
        """numpy.ndarray: The modes of the result data."""
        return self._nddata[D_MODES].to_numpy()

    @property
    def times_(self):
        """numpy.ndarray: The time values of the result data."""
        return self._nddata[D_TIMES].to_numpy()

    @property
    def positions_(self):
        """numpy.ndarray: The position values of the result data."""
        return self._nddata[D_POSITIONS].to_numpy()

    @property
    def positions_coordinates_(self):
        """numpy.ndarray: The position coordinates of the result data."""
        return self._nddata[D_POSITIONS_COORDINATES].to_numpy()

    pcoords_ = positions_coordinates_

    # UTILS ===================================================================

    def __repr__(self):
        """Return a string representation of the NDResult object."""
        cls_name = type(self).__name__
        mname = self.mname
        modes = self.modes_
        _, times, pos, pos_coords = self._nddata.shape
        causes = False if self.causes_ is None else self.causes_

        return (
            f"<{cls_name} '{mname}', modes={modes!s}, "
            f"times={times}, positions={pos}, "
            f"positions_coordinates={pos_coords}, causes={causes}>"
        )

    # ACCESSORS ===============================================================

    @property
    def plot(self):
        """ResultPlotter: Plot accessor for the NDResult object."""
        if not hasattr(self, "_plot"):
            self._plot = ResultPlotter(self)
        return self._plot

    @property
    def stats(self):
        """ResultStatsAccessor: Stats accessor for the NDResult object."""
        if not hasattr(self, "_stats"):
            self._stats = ResultStatsAccessor(self)
        return self._stats

    # DF BY DIMENSION =========================================================
    def _coherce_filters(self, flt, defaults, dim_name):
        """Coerce filters for a given dimension.

        Parameters
        ----------
        flt : str, int, float, numpy.number, Iterable, or None
            The filter value(s) for the dimension.
        defaults : Iterable
            The default values for the dimension.
        dim_name : str
            The name of the dimension.

        Returns
        -------
        list
            The coerced filter values.

        Raises
        ------
        ValueError
            If any filter value is not found in the defaults.

        """
        if flt is None:
            return list(defaults)

        if isinstance(flt, (str, int, float, np.number)):
            flt = [flt]
        elif isinstance(flt, Iterable):
            flt = list(flt)

        diff = set(flt).difference(defaults)
        if diff:
            diff_str = ", ".join(map(repr, diff))
            raise ValueError(f"{dim_name} {diff_str} not found")

        return flt

    def _dim_as_dataframe(self, flt, dim_name):
        """Convert a dimension to a pandas DataFrame.

        Parameters
        ----------
        flt : list
            The filter values for the dimension.
        dim_name : str
            The name of the dimension.

        Returns
        -------
        pandas.DataFrame
            The dimension as a DataFrame.

        """
        xa, dfs = self._nddata.sel({dim_name: flt}), []

        for gname, group in xa.groupby(dim_name):
            partial_df = group.to_dataframe(name=gname)

            partial_df = (
                partial_df.drop(dim_name, axis="columns")
                if dim_name in partial_df.columns
                else partial_df.droplevel(dim_name)
            )

            dfs.append(partial_df)

        df = pd.concat(dfs, axis="columns")
        df.columns.name = dim_name
        return df

    def get_modes(self, *, include=None):
        """Get the modes of the result data as a DataFrame.

        Parameters
        ----------
        include : str, int, float, numpy.number, Iterable, or None, optional
            The modes to include in the DataFrame. If None, all modes are
            included.

        Returns
        -------
        pandas.DataFrame
            The modes as a DataFrame.

        """
        flt = self._coherce_filters(include, self.modes_, D_MODES)
        df = self._dim_as_dataframe(flt, D_MODES)
        return df

    def get_times(self, *, include=None):
        """Get the time values of the result data as a DataFrame.

        Parameters
        ----------
        include : str, int, float, numpy.number, Iterable, or None, optional
            The time values to include in the DataFrame. If None, all time
            values are included.

        Returns
        -------
        pandas.DataFrame
            The time values as a DataFrame.

        """
        flt = self._coherce_filters(include, self.times_, D_TIMES)
        df = self._dim_as_dataframe(flt, D_TIMES)
        return df

    def get_positions(self, *, include=None):
        """Get the position values of the result data as a DataFrame.

        Parameters
        ----------
        include : str, int, float, numpy.number, Iterable, or None, optional
            The position values to include in the DataFrame. If None, all
            position values are included.

        Returns
        -------
        pandas.DataFrame
            The position values as a DataFrame.

        """
        flt = self._coherce_filters(include, self.positions_, D_POSITIONS)
        df = self._dim_as_dataframe(flt, D_POSITIONS)
        return df

    def get_positions_coordinates(self, *, include=None):
        """Get the position coordinates of the result data as a DataFrame.

        Parameters
        ----------
        include : str, int, float, numpy.number, Iterable, or None, optional
            The position coordinates to include in the DataFrame. If None, all
            position coordinates are included.

        Returns
        -------
        pandas.DataFrame
            The position coordinates as a DataFrame.

        """
        flt = self._coherce_filters(
            include, self.positions_coordinates_, D_POSITIONS_COORDINATES
        )
        df = self._dim_as_dataframe(flt, D_POSITIONS_COORDINATES)
        return df

    get_pcoords = get_positions_coordinates

    # IO ======================================================================

    def to_xarray(self):
        """Return a copy of the result data as an xarray.DataArray."""
        return self._nddata.copy()

    def to_dict(self):
        """Convert the NDResult object to a dictionary.

        Returns
        -------
        dict
            The NDResult object as a dictionary.

        """
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
            "run_parameters": self.run_parameters.to_dict(),
            "extra": self.extra_.to_dict(),
            "nddata": self.to_xarray(),
        }

    def to_ndr(self, path_or_stream, metadata=None, **kwargs):
        """Store the NDResult object in NMSI Result (NDR) format.

        Parameters
        ----------
        path_or_stream : str or file-like object
            The path or file-like object to store the NDR data.
        metadata : dict, optional
            Additional metadata to include in the NDR data.
        **kwargs
            Additional keyword arguments to pass to the NDR storage function.

        """
        from ...io import store_ndresult  # noqa

        store_ndresult(path_or_stream, self, metadata=metadata, **kwargs)

    # DTYPES HELPS ============================================================

    def astype(self, dtype, *, attributes=None):
        """Return a copy of the NDResult object with the specified data type.

        Parameters
        ----------
        dtype : data type
            The data type to convert the NDResult object to.
        attributes : list of str, optional
            The names of the attributes to convert. If None, all attributes

        Returns
        -------
        NDResult
            The NDResult object with the specified data type.

        """
        kwargs = self.to_dict()
        for k, v in kwargs.items():
            if attributes is None or k in attributes:
                kwargs[k] = ddtype_tools.deep_astype(v, dtype)

        cls = type(self)  # get the class
        return cls(**kwargs)  # create a new instance

    def deep_dtypes(self, *, max_deep=2, memory_usage=False):
        """Returns the deep data types of the object.

        Parameters
        ----------
        max_deep : int, optional
            The maximum depth to traverse the object. Defaults to 2.
        memory_usage : bool, optional
            If True, return the memory usage of the object. Defaults to False.

        Returns
        -------
        dict
            The deep data types of the object.

        """
        ddtypes = ddtype_tools.deep_dtypes(
            self.to_dict(),
            root="ndresult",
            max_deep=max_deep,
            memory_usage=memory_usage,
        )
        # ddtypes = ddtypes[0] if memory_usage else ddtypes
        return ddtypes["ndresult"][1]

    def dtypes(self, *, memory_usage=False):
        """pd.DataFrame containing the data types of each attribute in the \
        NDResult object."""
        ddtypes = self.deep_dtypes(max_deep=2, memory_usage=memory_usage)
        dtypes = []
        for attr, obj_info in ddtypes.items():
            obj_type, obj_dtype = obj_info[:2]
            dtype = (
                obj_dtype if ddtype_tools.single_dtype_class(obj_type) else "-"
            )

            mem = obj_info[-1].hsize if memory_usage else "?"

            dtypes.append(
                {
                    "Attribute": attr,
                    "Type": obj_type,
                    "DType": dtype,
                    "Size": mem,
                }
            )

        dtypes_df = pd.DataFrame(dtypes)
        dtypes_df.set_index("Attribute", inplace=True)
        dtypes_df.name = "dtypes"

        return dtypes_df
