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

import contextlib
import datetime as dt
import platform
import sys
import json
from collections import namedtuple
import zipfile

import numpy as np

from pandas.io import common as pd_io_common

import xarray as xa

from . import core, VERSION

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_ENGINE = "h5netcdf"

_FORMAT =""

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


class _Keys:
    SKN_METADATA_KEY = "__skneuromsi__"
    UTC_TIMESTAMP_KEY = "utc_timestamp"
    OBJ_TYPE_KEY = "object_type"
    OBJ_KWARGS_KEY = "object_kwargs"
    OBJ_SIZE_KEY = "object_size"
    EXTRA_METADATA_KEYS = "extra"


class _ObjTypes:

    NDRESULT_TYPE = "ndresult"
    NDCOLLETION_TYPE = "ndcollection"


# =============================================================================
# IO HELPERS
# =============================================================================


@contextlib.contextmanager
def _get_buffer(buf, mode):
    # Based on https://github.com/pandas-dev/pandas/blob/
    # 48f5a961cb58b535e370c5688a336bc45493e404/
    # pandas/io/formats/format.py#L1180

    if hasattr(buf, "write"):
        yield buf
    elif isinstance(buf, str):
        pd_io_common.check_parent_directory(str(buf))
        with open(buf, mode) as fp:
            yield fp
    else:
        raise TypeError("buf is not a file name and it has no write method")


# =============================================================================
# JSON CONVERTERS
# =============================================================================


class NDResultJSONEncoder(json.JSONEncoder):
    _CONVERTERS = {
        tuple: list,
        set: list,
        frozenset: list,
        dt.datetime: dt.datetime.isoformat,
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


# =============================================================================
# METADATA MERGER
# =============================================================================

def _prepare_ndr_metadata(
    size, obj_type, obj_kwargs, utc_timestamp, extra_metadata
):

    # prepare metadata with the default values, time and custom metadata
    nc_metadata = _DEFAULT_METADATA.copy()
    nc_metadata.update(
        {
            _Keys.UTC_TIMESTAMP_KEY: utc_timestamp,
            _Keys.OBJ_TYPE_KEY: obj_type,
            _Keys.OBJ_KWARGS_KEY: obj_kwargs,
            _Keys.EXTRA_METADATA_KEYS: extra_metadata,
        }
    )

    return nc_metadata


# =============================================================================
# NDRESULT IO
# =============================================================================


def to_ndr(path_or_stream, ndresult, metadata=None, **kwargs):
    if not isinstance(ndresult, core.NDResult):
        raise TypeError(f"'ndresult' must be an instance of {core.NDResult!r}")

    # convert the ndresult to dict and extract the xarray
    ndresult_kwargs = ndresult.to_dict()
    nddata = ndresult_kwargs.pop("nddata")

    # timestamp
    now = dt.datetime.utcnow()

    ndr_metadata = _prepare_ndr_metadata(
        size=None,
        obj_type=_ObjTypes.NDRESULT_TYPE,
        obj_kwargs=ndresult_kwargs,
        utc_timestamp=now,
        extra_metadata=metadata or {},
    )

    with zipfile.ZipFile(path_or_stream, "w", zipfile.ZIP_DEFLATED) as zfp:

        json_buff = json.dumps(ndr_metadata, cls=NDResultJSONEncoder)
        zfp.writestr(METADATA_FILENAME, json_buff)

        nddata_buff = io.BytesIO()
        nddata.to_netcdf(nddata_buff, group=None, **kwargs)
        zfp.writestr(NDDATA_NC_FILENAME, nddata_buff.getvalue())

        del nddata_buff


def read_ndr(path_or_stream, **kwargs):
    with _get_buffer(path_or_stream, "rb") as fp:
        # For some reason if i add the _DEFAULT_ENGINE here the code not work
        nddata = xa.open_dataarray(fp, group=None, **kwargs)

    # extract and remove all the sknmsi metadata
    nc_metadata = json.loads(nddata.attrs.pop(_Keys.SKN_METADATA_KEY))

    # type verification
    obj_type = nc_metadata.pop(_Keys.OBJ_TYPE_KEY)
    if obj_type != _ObjTypes.NDRESULT_TYPE:
        raise ValueError(
            f"'ndr' files must have 'object_type={_ObjTypes.NDRESULT_TYPE}'. "
            f"Found {obj_type!r}"
        )

    ndresult_kwargs = nc_metadata[_Keys.OBJ_KWARGS_KEY]

    return core.NDResult(nddata=nddata, **ndresult_kwargs)


# =============================================================================
# NDCOLLECTION
# =============================================================================


def ndcollection_to_netcdf(
    path_or_stream, ndcollection, metadata=None, **kwargs
):
    length = len(ndcollection)
    empty_da = xa.DataArray()
