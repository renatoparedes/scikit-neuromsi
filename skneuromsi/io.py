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

"""Implementation of I/O for skneuromsi.

This module provides functions for storing and loading NDResult and
NDResultCollection objects to and from files or file-like objects using a
zip-based format.

The main functions provided are:

- store_ndresult: Store a single NDResult object to a file or stream.
- open_ndresult: Open a single NDResult object from a file or stream.
- store_ndrcollection: Store an NDResultCollection to a file or stream.
- open_ndrcollection: Open an NDResultCollection from a file or stream.

The NDResult and NDResultCollection objects are serialized using a combination
of JSON (for metadata) and NetCDF (for the underlying nddata). The resulting
files are zip archives containing the serialized metadata and data.

The module also includes several helper functions and classes for serializing
and deserializing NDResult and NDResultCollection objects, as well as for
managing metadata and storage backends.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import json
import platform
import sys
import zipfile

import numpy as np

from tqdm.auto import tqdm

import xarray as xa

from . import VERSION, core
from .utils import storages


# =============================================================================
# CONSTANTS
# =============================================================================

#: Default metadata
_DEFAULT_METADATA = {
    "skneuromsi": VERSION,
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
# JSON CONVERTERS
# =============================================================================


class NDResultJSONEncoder(json.JSONEncoder):
    """JSON encoder for NDResult objects and related data types.

    This encoder extends the default JSON encoder to support serializing
    additional data types commonly used with NDResult objects, such as
    tuples, sets, frozensets, datetime objects, and NumPy data types.

    """

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
        """Override the default method to serialize additional types.

        This method gets called by the json.JSONEncoder superclass when
        encountering an object that is neither a dict, list, str, nor any other
        JSON-encodable type. This implementation extends support to serialize
        tuples, sets, frozensets, datetime objects, and NumPy data types.
        If the object is not serializable, a TypeError is raised to signal
        that the default serialization method cannot handle this object type.

        Parameters
        ----------
        obj : object
            The object to serialize.

        Returns
        -------
        serializable : object
            A JSON-encodable version of the object.

        Raises
        ------
        TypeError
            If the object is not serializable.

        """
        for nptype, converter in self._CONVERTERS.items():
            if isinstance(obj, nptype):
                return converter(obj)
        return super(NDResultJSONEncoder, self).default(obj)


# =============================================================================
# METADATA MERGER
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
    ndr_metadata_json = json.dumps(
        ndr_metadata, cls=NDResultJSONEncoder, indent=2
    )

    return ndr_nddata_nc, ndr_metadata_json


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


def _read_nd_results_into_storage(zip_fp, storage, tqdm_cls):
    """Read NDResult objects from a zip file into a storage backend.

    Parameters
    ----------
    zip_fp : zipfile.ZipFile
        The zip file containing the NDResult data.
    storage : StorageBackend
        The storage backend to load the NDResults into.
    tqdm_cls : type
        The progress bar class to use (or None to disable progress bars).

    Returns
    -------
    StorageBackend
        The storage backend populated with the loaded NDResult objects.

    """
    indexes = range(len(storage))
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

        storage[idx] = ndresult

        del ndr_metadata, nddata, ndresult_kwargs, ndresult


# =============================================================================
# NDCOLLECTION
# =============================================================================


def store_ndrcollection(
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
    tqdm_cls : type, optional
        The progress bar class to use (or None to disable progress bars).
    **kwargs
        Additional keyword arguments to pass to zipfile.ZipFile.

    Raises
    ------
    TypeError
        If `ndrcollection` is not an instance of NDResultCollection.

    """
    if not isinstance(ndrcollection, core.NDResultCollection):
        raise TypeError(
            "'ndrcollection' must be an instance "
            f"of {core.NDResultCollection!r}"
        )

    # default parameters for zipfile
    kwargs.setdefault("compression", _Compression.COMPRESSION)
    kwargs.setdefault("compresslevel", _Compression.COMPRESS_LEVEL)

    # extract the data from the object
    ndrcollection_kwargs = ndrcollection.to_dict()
    ndresults = ndrcollection_kwargs.pop("ndresults")

    # timestamp
    timestamp = dt.datetime.utcnow()

    # collection of metadata
    ndc_metadata = _prepare_ndc_metadata(
        size=len(ndresults),
        obj_type=_ObjTypes.NDCOLLETION_TYPE,
        obj_kwargs=ndrcollection_kwargs,
        utc_timestamp=timestamp,
        extra_metadata=metadata or {},
    )

    with zipfile.ZipFile(path_or_stream, "w", **kwargs) as zip_fp:
        # write the collection metadata.json
        ndc_metadata_json = json.dumps(
            ndc_metadata, cls=NDResultJSONEncoder, indent=2
        )
        zip_fp.writestr(_ZipFileNames.METADATA, ndc_metadata_json)

        if tqdm_cls:
            ndresults = tqdm_cls(iterable=ndresults, desc="Writing ndresults")

        # write every ndresult
        for idx, ndresult in enumerate(ndresults):
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


def open_ndrcollection(
    path_or_stream,
    *,
    storage="directory",
    storage_kws=None,
    expected_size=None,
    tqdm_cls=tqdm,
    **kwargs,
):
    """Retrieve an NDResultCollection from a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to read the NDResultCollection from.
    storage : str or StorageBackend, optional
        The storage backend to use for the loaded NDResults.
    storage_kws : dict, optional
        Additional keyword arguments to pass to the storage backend
        constructor.
    expected_size : int, optional
        The expected number of NDResults in the collection
        (or None to disable checking).
    tqdm_cls : type, optional
        The progress bar class to use (or None to disable progress bars).
    **kwargs
        Additional keyword arguments to pass to zipfile.ZipFile.

    Returns
    -------
    NDResultCollection
        The loaded NDResultCollection object.

    Raises
    ------
    ValueError
        If the number of loaded NDResults does not match `expected_size`.

    """
    # default parameters for storage_kws
    storage_kws = {} if storage_kws is None else dict(storage_kws)

    with zipfile.ZipFile(path_or_stream, "r", **kwargs) as zip_fp:
        # open the collection metadata
        with zip_fp.open(_ZipFileNames.METADATA) as fp:
            ndc_metadata = json.load(fp)

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
        tag = ndcollection_kwargs.get("name", "<UNKNOW>")

        # open the storage and read the entire ndresults
        with storages.storage(
            storage, size=size, tag=tag, **storage_kws
        ) as results:
            _read_nd_results_into_storage(
                zip_fp=zip_fp, storage=results, tqdm_cls=tqdm_cls
            )

        # store the results inside the ndr collection
        ndr_collection = core.NDResultCollection(
            results=results, tqdm_cls=tqdm_cls, **ndcollection_kwargs
        )

        return ndr_collection


# =============================================================================
# ND RESULT
# =============================================================================


def store_ndresult(path_or_stream, ndresult, *, metadata=None, **kwargs):
    """Store a single NDResult object to a file or stream.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to write the NDResult to.
    ndresult : NDResult
        The NDResult object to store.
    metadata : dict, optional
        Additional metadata to include in the output file.
    **kwargs
        Additional keyword arguments to pass to zipfile.ZipFile.

    Raises
    ------
    TypeError
        If `ndresult` is not an instance of NDResult.

    """
    if not isinstance(ndresult, core.NDResult):
        raise TypeError(f"'ndresult' must be an instance of {core.NDResult!r}")

    ndrcollection = core.NDResultCollection(
        "NDResult", [ndresult], tqdm_cls=None
    )
    store_ndrcollection(
        path_or_stream, ndrcollection, metadata=metadata, **kwargs
    )


def open_ndresult(path_or_stream, **kwargs):
    """Open a single NDResult object from a file or stream.

    This function loads a single NDResult object that was previously stored
    using the `store_ndresult` function. It expects the input file or stream to
    be a zip archive containing the serialized metadata and data for the
    NDResult.

    Under the hood, this function uses `open_ndrcollection` to load the
    NDResult as an NDResultCollection with a single element, and then returns
    that element.

    Parameters
    ----------
    path_or_stream : str or file-like object
        The file path or stream to read the NDResult from.
    **kwargs
        Additional keyword arguments to pass to `open_ndrcollection`.

    Returns
    -------
    NDResult
        The loaded NDResult object.

    See Also
    --------
    store_ndresult : Store a single NDResult object to a file or stream.
    open_ndrcollection : Open an NDResultCollection from a file or stream.

    Examples
    --------
    >>> import skneuromsi.io as ndio
    >>> ndresult = ndio.open_ndresult('path/to/ndresult.ndr')

    """
    ndr_collection = open_ndrcollection(
        path_or_stream,
        storage="single_process",
        expected_size=1,
        tqdm_cls=None,
        **kwargs,
    )
    return ndr_collection[0]
