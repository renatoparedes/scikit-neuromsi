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

"""Implementation of I/O for skneuromsi.

This module provides functions for storing and loading NDResult and
NDResultCollection objects to and from files or file-like objects using a
zip-based format.

The NDResult and NDResultCollection objects are serialized using a combination
of JSON (for metadata) and NetCDF (for the underlying nddata). The resulting
files are zip archives containing the serialized metadata and data.


"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import json
import platform
import sys
import zipfile

from tqdm.auto import tqdm

import xarray as xa

from . import core, ndcollection
from .utils import custom_json


# =============================================================================
# CONSTANTS
# =============================================================================

#: Default metadata
_DEFAULT_METADATA = {
    "skneuromsi": ".".join(map(str, core.VERSION)),
    "authors": "Paredes, Cabral & Seriès",
    "author_email": "paredesrenato92@gmail.com",
    "affiliation": [
        (
            "Cognitive Science Group, "
            "Instituto de Investigaciones Psicológicas, "
            "Facultad de Psicología - UNC-CONICET. "
            "Córdoba, Córdoba, Argentina."
        ),
        (
            "Department of Psychology, "
            "Pontifical Catholic University of Peru, Lima, Peru."
        ),
        (
            "The University of Edinburgh, School of Informatics, "
            "Edinburgh, United Kingdom."
        ),
        (
            "Gerencia De Vinculacion Tecnológica "
            "Comisión Nacional de Actividades Espaciales (CONAE), "
            "Falda del Cañete, Córdoba, Argentina."
        ),
        (
            "Instituto De Astronomía Teorica y Experimental - "
            "Observatorio Astronómico Córdoba (IATE-OAC-UNC-CONICET), "
            "Cordoba, Argentina."
        ),
    ],
    "url": "https://github.com/renatoparedes/scikit-neuromsi",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
    "format_version": 0.1,
}


class _Keys:
    """Constants for keys used in metadata dictionaries."""

    UTC_TIMESTAMP_KEY = "utc_timestamp"
    OBJ_TYPE_KEY = "object_type"
    OBJ_KWARGS_KEY = "object_kwargs"
    OBJ_SIZE_KEY = "object_size"
    EXTRA_METADATA_KEYS = "extra"


class _ZipFileNames:
    """Constants for filenames used within zip archives."""

    METADATA = "metadata.json"
    NDDATA = "nddata.nc"


class _ObjTypes:
    """Constants for object type identifiers."""

    NDRESULT_TYPE = "ndresult"
    NDCOLLETION_TYPE = "ndcollection"


class _Compression:
    """Constants for compression settings."""

    COMPRESSION = zipfile.ZIP_DEFLATED
    COMPRESS_LEVEL = 9


# =============================================================================
# STORE
# =============================================================================


def _prepare_ndc_metadata(
    size, obj_type, obj_kwargs, utc_timestamp, extra_metadata
):
    """Prepare metadata for an NDResultCollection.

    Parameters
    ----------
    size : int
        The number of NDResult objects in the collection.
    obj_type : str
        The type of the object being serialized
        (e.g., 'ndresult' or 'ndcollection').
    obj_kwargs : dict
        Additional keyword arguments to include in the metadata.
    utc_timestamp : datetime.datetime
        The UTC timestamp to include in the metadata.
    extra_metadata : dict
        Additional custom metadata to include.

    Returns
    -------
    dict
        The prepared metadata dictionary.

    """
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


def _ndr_split_and_serialize(ndresult):
    """Split an NDResult into metadata and data, and serialize them.

    Parameters
    ----------
    ndresult : NDResult
        The NDResult object to split and serialize.

    Returns
    -------
    tuple
        A tuple containing the serialized NDResult data (as NetCDF bytes)
        and the serialized NDResult metadata (as a JSON string).

    """
    # convert the ndresult to dict and extract the xarray
    ndresult_kwargs = ndresult.to_dict()
    ndr_nddata = ndresult_kwargs.pop("nddata")

    ndr_metadata = {
        _Keys.OBJ_TYPE_KEY: _ObjTypes.NDRESULT_TYPE,
        _Keys.OBJ_KWARGS_KEY: ndresult_kwargs,
    }

    ndr_nddata_nc = ndr_nddata.to_netcdf(None)
    ndr_metadata_json = custom_json.dumps(ndr_metadata, indent=2)

    return ndr_nddata_nc, ndr_metadata_json


def _mk_ndr_in_zip_paths(idx):
    """Generate the zip paths for an NDResult at a given index.

    Parameters
    ----------
    idx : int
        The index of the NDResult.

    Returns
    -------
    tuple
        A tuple containing the metadata filename and NDResult data filename.

    """
    ndr_metadata_filename = f"ndr_{idx}/{_ZipFileNames.METADATA}"
    ndr_nddata_filename = f"ndr_{idx}/{_ZipFileNames.NDDATA}"
    return ndr_metadata_filename, ndr_nddata_filename


# API STORE ===================================================================


def store_ndresults_collection(
    path_or_stream, ndrcollection, *, metadata=None, tqdm_cls=tqdm, **kwargs
):
    """Store an NDResultCollection to a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to write the NDResultCollection to.
    ndrcollection : NDResultCollection
        The NDResultCollection object to store.
    metadata : dict, optional
        Additional metadata to include in the output file.
    **kwargs
        Additional keyword arguments to pass to zipfile.ZipFile.

    Raises
    ------
    TypeError
        If `ndrcollection` is not an instance of NDResultCollection.

    """
    if not isinstance(ndrcollection, ndcollection.NDResultCollection):
        raise TypeError(
            "'ndrcollection' must be an instance "
            f"of {ndcollection.NDResultCollection!r}"
        )

    # default parameters for zipfile
    kwargs.setdefault("compression", _Compression.COMPRESSION)
    kwargs.setdefault("compresslevel", _Compression.COMPRESS_LEVEL)

    # timestamp
    timestamp = dt.datetime.utcnow()

    # collection of metadata
    ndc_metadata = _prepare_ndc_metadata(
        size=len(ndrcollection),
        obj_type=_ObjTypes.NDCOLLETION_TYPE,
        obj_kwargs={"name": ndrcollection.name},
        utc_timestamp=timestamp,
        extra_metadata=metadata or {},
    )

    # serialize metadataa
    ndc_metadata_json = custom_json.dumps(ndc_metadata, indent=2)

    if tqdm_cls:
        ndrcollection = tqdm_cls(
            ndrcollection,
            total=len(ndrcollection),
            desc=f"Saving '{str(path_or_stream)}'",
        )

    with zipfile.ZipFile(path_or_stream, "w", **kwargs) as zip_fp:
        # write every ndresult
        for idx, ndresult in enumerate(ndrcollection):
            # determine the directory
            ndr_metadata_filename, ndr_nddata_filename = _mk_ndr_in_zip_paths(
                idx
            )

            # serielize the ndresult
            ndr_nddata_nc, ndr_metadata_json = _ndr_split_and_serialize(
                ndresult
            )

            # write
            zip_fp.writestr(ndr_nddata_filename, ndr_nddata_nc)
            zip_fp.writestr(ndr_metadata_filename, ndr_metadata_json)

            del ndresult, ndr_nddata_nc, ndr_metadata_json

        # write the collection metadata.json
        zip_fp.writestr(_ZipFileNames.METADATA, ndc_metadata_json)


def store_ndresult(path_or_stream, ndresult, *, metadata=None, **kwargs):
    """
    Store a single NDResult object to a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to write the NDResult to.
    ndresult : NDResult
        The NDResult object to store.
    metadata : dict, optional
        Additional metadata to include in the output file.
    **kwargs
        Additional keyword arguments to pass to store_ndrcollection.

    Raises
    ------
    TypeError
        If `ndresult` is not an instance of NDResult.
    """
    if not isinstance(ndresult, core.NDResult):
        raise TypeError(f"'ndresult' must be an instance of {core.NDResult!r}")

    cls_name = type(ndresult).__name__
    ndrcollection = ndcollection.NDResultCollection.from_ndresults(
        cls_name, [ndresult]
    )

    store_ndresults_collection(
        path_or_stream,
        ndrcollection,
        metadata=metadata,
        tqdm_cls=None,
        **kwargs,
    )


# =============================================================================
# READ
# =============================================================================


def _check_object_type(obj_type, expected):
    """Check that an object type matches the expected value.

    Parameters
    ----------
    obj_type : str
        The object type to check.
    expected : str
        The expected object type.

    Raises
    ------
    ValueError
        If the object type does not match the expected value.

    """
    if obj_type != expected:
        raise ValueError(f"'object_type' != {expected!r}. Found {obj_type!r}")


def _generate_ndresults(*, zip_fp, size, tqdm_cls):
    """Read NDResult objects from a zip file into a storage backend."""
    indexes = range(size)

    if tqdm_cls:
        indexes = tqdm_cls(iterable=indexes, desc="Reading ndresults")

    for idx in indexes:
        # determine the directory
        ndr_metadata_filename, ndr_nddata_filename = _mk_ndr_in_zip_paths(idx)

        with zip_fp.open(ndr_metadata_filename) as fp:
            ndr_metadata = json.load(fp)

        obj_type = ndr_metadata.pop(_Keys.OBJ_TYPE_KEY)
        _check_object_type(obj_type, _ObjTypes.NDRESULT_TYPE)

        with zip_fp.open(ndr_nddata_filename) as fp:
            nddata = xa.open_dataarray(fp).compute()

        ndresult_kwargs = ndr_metadata[_Keys.OBJ_KWARGS_KEY]
        ndresult = core.NDResult(nddata=nddata, **ndresult_kwargs)

        yield ndresult


# API READ ====================================================================


def open_ndresults_collection(
    path_or_stream,
    *,
    compression_params=core.DEFAULT_COMPRESSION_PARAMS,
    expected_size=None,
    tqdm_cls=tqdm,
    **kwargs,
):
    """Retrieve an NDResultCollection from a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to read the NDResultCollection from.
    compression_params : dict, optional
        Compression parameters for the NDResultCollection.
    expected_size : int, optional
        The expected number of NDResult objects in the collection.
    tqdm_cls : callable, optional
        The tqdm class to use for progress bars.
    **kwargs
        Additional keyword arguments to pass to zipfile.ZipFile.

    Returns
    -------
    NDResultCollection
        The retrieved NDResultCollection object.

    Raises
    ------
    ValueError
        If the expected size doesn't match the actual size of the collection.

    """
    with zipfile.ZipFile(path_or_stream, "r", **kwargs) as zip_fp:
        # open the collection metadata
        with zip_fp.open(_ZipFileNames.METADATA) as fp:
            ndc_metadata = custom_json.load(fp)

        # validate the object type
        obj_type = ndc_metadata.pop(_Keys.OBJ_TYPE_KEY)
        _check_object_type(obj_type, _ObjTypes.NDCOLLETION_TYPE)

        # extract the extra arguments needed to create an dncollection
        ndcollection_kwargs = ndc_metadata[_Keys.OBJ_KWARGS_KEY]

        # retrieve the collection size and check if the size is correct
        size = ndc_metadata[_Keys.OBJ_SIZE_KEY]

        if expected_size is not None and size != int(expected_size):
            raise ValueError(
                f"{str(path_or_stream)}: Expected {expected_size} "
                f"results, but {size} were found"
            )

        # create the tag for the storage
        tag = ndcollection_kwargs.pop("name", "<UNKNOW>")

        nd_results_gen = _generate_ndresults(
            zip_fp=zip_fp, size=size, tqdm_cls=tqdm_cls
        )

        # store the results inside the ndr collection
        ndr_collection = ndcollection.NDResultCollection.from_ndresults(
            name=tag,
            results=nd_results_gen,
            tqdm_cls=tqdm_cls,
            compression_params=compression_params,
            **ndcollection_kwargs,
        )

        return ndr_collection


def open_ndresult(path_or_stream, **kwargs):
    """
    Open a single NDResult object from a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to read the NDResult from.
    **kwargs
        Additional keyword arguments to pass to open_ndrcollection.

    Returns
    -------
    NDResult
        The retrieved NDResult object.
    """
    ndr_collection = open_ndresults_collection(
        path_or_stream,
        expected_size=1,
        compression_params=None,
        tqdm_cls=None,
        **kwargs,
    )
    return ndr_collection[0]


# SHORTCUTS ===================================================================

to_ndr = store_ndresult
read_ndr = open_ndresult
to_ndc = store_ndresults_collection
read_ndc = open_ndresults_collection
