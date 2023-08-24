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
# FUNCTIONALITIES
# =============================================================================


def to_sknmsi_nc(path_or_stream, ndresult, metadata=None, **kwargs):

    # prepare metadata
    nc_metadata = _DEFAULT_METADATA.copy()
    nc_metadata[_UTC_TIMESTAMP_KEY] = dt.datetime.utcnow().isoformat()
    nc_metadata[_NDRESULT_KWARGS_KEY] = {
        "mname": ndresult.mname,
        "mtype": ndresult.mtype,
        "output_mode": ndresult.output_mode,
        "nmap": ndresult.nmap_,
        "time_range": ndresult.time_range.tolist(),
        "position_range": ndresult.position_range.tolist(),
        "time_res": ndresult.time_res,
        "position_res": ndresult.position_res,
        "causes": ndresult.causes_,
        "run_params": ndresult.run_params.to_dict(),
        "extra": ndresult.extra_.to_dict(),
    }
    nc_metadata.update(metadata or {})
    nc_metadata_json = json.dumps(nc_metadata)

    # convert the data to xarray
    nddata = ndresult.to_xarray()
    nddata.attrs[_METADATA_KEY] = nc_metadata_json

    return nddata.to_netcdf(
        path_or_stream, engine="netcdf4", group=None, **kwargs
    )


def read_sknmsi_nc(path_or_stream, **kwargs):
    nddata = xa.open_dataarray(
        path_or_stream, engine="netcdf4", group=None, **kwargs
    )
    nc_metadata = json.loads(nddata.attrs[_METADATA_KEY])
    del nddata.attrs[_METADATA_KEY]

    ndresult_kwargs = nc_metadata[_NDRESULT_KWARGS_KEY]

    return core.NDResult(nddata=nddata, **ndresult_kwargs)
