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

"""Implementation of I/O for skneuromsi."""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import json
import platform
import sys
import zipfile

import numpy as np

import xarray as xa

from . import VERSION, cmp, core

# =============================================================================
# CONSTANTS
# =============================================================================


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
    UTC_TIMESTAMP_KEY = "utc_timestamp"
    OBJ_TYPE_KEY = "object_type"
    OBJ_KWARGS_KEY = "object_kwargs"
    OBJ_SIZE_KEY = "object_size"
    EXTRA_METADATA_KEYS = "extra"


class _ZipFileNames:
    METADATA = "metadata.json"
    NDDATA = "nddata.nc"


class _ObjTypes:

    NDRESULT_TYPE = "ndresult"
    NDCOLLETION_TYPE = "ndcollection"


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


def _prepare_metadata(
    size, obj_type, obj_kwargs, utc_timestamp, extra_metadata
):

    # prepare metadata with the default values, time and custom metadata
    nc_metadata = _DEFAULT_METADATA.copy()
    nc_metadata.update(
        {
            _Keys.OBJ_SIZE_KEY: size,
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


def _ndr_split_and_serialize(ndresult, timestamp, extra_metadata):
    # convert the ndresult to dict and extract the xarray
    ndresult_kwargs = ndresult.to_dict()
    ndr_nddata = ndresult_kwargs.pop("nddata")

    ndr_metadata = _prepare_metadata(
        size=None,
        obj_type=_ObjTypes.NDRESULT_TYPE,
        obj_kwargs=ndresult_kwargs,
        utc_timestamp=timestamp,
        extra_metadata=extra_metadata or {},
    )

    ndr_nddata_nc = ndr_nddata.to_netcdf(None, group=None)
    ndr_metadata_json = json.dumps(
        ndr_metadata, cls=NDResultJSONEncoder, indent=2
    )

    return ndr_nddata_nc, ndr_metadata_json


def to_ndr(path_or_stream, ndresult, metadata=None):
    if not isinstance(ndresult, core.NDResult):
        raise TypeError(f"'ndresult' must be an instance of {core.NDResult!r}")

    # timestamp
    timestamp = dt.datetime.utcnow()
    ndr_nddata_nc, ndr_metadata_json = _ndr_split_and_serialize(
        ndresult, timestamp, metadata
    )

    with zipfile.ZipFile(path_or_stream, "w", zipfile.ZIP_DEFLATED) as zfp:
        zfp.writestr(_ZipFileNames.NDDATA, ndr_nddata_nc)
        zfp.writestr(_ZipFileNames.METADATA, ndr_metadata_json)


def read_ndr(path_or_stream, **kwargs):
    with zipfile.ZipFile(path_or_stream, "r", zipfile.ZIP_DEFLATED) as zfp:
        with zfp.open(_ZipFileNames.METADATA) as fp:
            ndr_metadata = json.load(fp)
        with zfp.open(_ZipFileNames.NDDATA) as fp:
            nddata = xa.open_dataarray(fp, group=None, **kwargs)

    obj_type = ndr_metadata.pop(_Keys.OBJ_TYPE_KEY)
    if obj_type != _ObjTypes.NDRESULT_TYPE:
        raise ValueError(
            f"'ndr' files must have 'object_type={_ObjTypes.NDRESULT_TYPE}'. "
            f"Found {obj_type!r}"
        )

    ndresult_kwargs = ndr_metadata[_Keys.OBJ_KWARGS_KEY]

    ndresult = core.NDResult(nddata=nddata, **ndresult_kwargs)
    return ndresult


# =============================================================================
# NDCOLLECTION
# =============================================================================


def to_ndc(path_or_stream, ndrcollection, metadata=None, **kwargs):
    if not isinstance(ndrcollection, cmp.NDResultCollection):
        raise TypeError(
            "'ndrcollection' must be an instance "
            f"of {cmp.NDResultCollection!r}"
        )

    ndrcollection_size = len(ndrcollection)
    ndrcollection_kwargs = {"name": ndrcollection.name}

    # timestamp
    timestamp = dt.datetime.utcnow()

    # collection metadata
    ndc_metadata = _prepare_metadata(
        size=ndrcollection_size,
        obj_type=_ObjTypes.NDCOLLETION_TYPE,
        obj_kwargs=ndrcollection_kwargs,
        utc_timestamp=timestamp,
        extra_metadata=metadata or {},
    )

    with zipfile.ZipFile(path_or_stream, "w", zipfile.ZIP_DEFLATED) as zfp:

        ndc_metadata_json = json.dumps(
            ndc_metadata, cls=NDResultJSONEncoder, indent=2
        )
        zfp.writestr(_ZipFileNames.METADATA, ndc_metadata_json)

        for idx in range(ndrcollection_size):
            ndresult = ndrcollection[idx]

            ndr_metadata_filename = f"ndr_{idx}/{_ZipFileNames.METADATA}"
            ndr_nddata_filename = f"ndr_{idx}/{_ZipFileNames.NDDATA}"

            ndr_nddata_nc, ndr_metadata_json = _ndr_split_and_serialize(
                ndresult, timestamp, metadata
            )

            zfp.writestr(ndr_nddata_filename, ndr_nddata_nc)
            zfp.writestr(ndr_metadata_filename, ndr_metadata_json)

            del ndresult
