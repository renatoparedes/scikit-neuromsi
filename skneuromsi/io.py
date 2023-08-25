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

"""Implementation of multisensory integration models."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import core, VERSION
import datetime as dt
import platform
import sys
import json

import numpy as np

import xarray as xa

# =============================================================================
# CONSTANTS
# =============================================================================

_METADATA_KEY = "__skneuromsi__"

_DEFAULT_METADATA = {
    "skneuromsi": VERSION,
    "author_email": "",
    "affiliation": "",
    "url": "https://github.com/",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
    "format_version": 0.1,
}


_UTC_TIMESTAMP_KEY = "utc_timestamp"

_NDRESULT_KWARGS_KEY = "ndresult_kwargs"


# =============================================================================
# NDRESULT IO
# =============================================================================


class NDResultJSONEncoder(json.JSONEncoder):
    _CONVERTERS = {
        tuple: list,
        set: list,
        frozenset: list,
        np.integer: int,
        np.floating: float,
        np.complexfloating: complex,
        np.bool_: bool,
        np.ndarray: np.ndarray.tolist,
    }

    def default(self, obj):
        for nptype, converter in self._CONVERTERS.items():
            if isinstance(obj, nptype):
                return converter(obj)
        return super(NDResultJSONEncoder, self).default(obj)


def ndresult_to_netcdf(path_or_stream, ndresult, metadata=None, **kwargs):

    # convert the ndresult to dict and extract the xarray
    ndresult_dict = ndresult.to_dict()
    nddata = ndresult_dict.pop("nddata")

    # prepare metadata
    nc_metadata = _DEFAULT_METADATA.copy()
    nc_metadata[_UTC_TIMESTAMP_KEY] = dt.datetime.utcnow().isoformat()
    nc_metadata[_NDRESULT_KWARGS_KEY] = ndresult_dict
    nc_metadata.update(metadata or {})
    nc_metadata_json = json.dumps(nc_metadata, cls=NDResultJSONEncoder)

    # convert the data to xarray
    nddata.attrs[_METADATA_KEY] = nc_metadata_json

    return nddata.to_netcdf(
        path_or_stream, engine="netcdf4", group=None, **kwargs
    )


def ndresult_from_netcdf(path_or_stream, **kwargs):
    nddata = xa.open_dataarray(
        path_or_stream, engine="netcdf4", group=None, **kwargs
    )
    nc_metadata = json.loads(nddata.attrs[_METADATA_KEY])
    del nddata.attrs[_METADATA_KEY]

    ndresult_kwargs = nc_metadata[_NDRESULT_KWARGS_KEY]

    return core.NDResult(nddata=nddata, **ndresult_kwargs)


# =============================================================================
# NDCOLLECTION
# =============================================================================
