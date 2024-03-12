#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

import abc
import atexit
import contextlib
import os
import shutil
import tempfile

import bidict
import joblib
import numpy as np

from .doctools import doc_inherit

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default storage directory
_DEFAULT_STORAGE_DIR = tempfile.mkdtemp(prefix="skneuromsi_")
atexit.register(shutil.rmtree, _DEFAULT_STORAGE_DIR)


# =============================================================================
# ABC CLASS
# =============================================================================
class StorageABC(metaclass=abc.ABCMeta):
    """Base class for all storages."""

    @abc.abstractmethod
    def lock(self):
        """Pone el storage en modo solo lectura."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx):
        """x.__getitem__(y) <==> x[y]."""
        pass

    @abc.abstractmethod
    def __setitem__(self, idx, v):
        """x.__setitem__(y, v) <==> x[y] = v."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """x.__len__() <==> len(x)."""
        pass

    def __delitem__(self, idx):
        """x.__delitem__(y) <==> del x[y]."""
        del self[idx]

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        stype = type(self).__name__
        filled = " ".join([("X" if e else "-") for e in self])
        repr_str = f"<{stype} [ {filled} ]>"
        return repr_str


# =============================================================================
# IN DISK
# =============================================================================


class DirectoryStorage(StorageABC):
    """DirectoryStorage class for storing data on disk.

    This stograge stores all data in a directory, keeping a numpy memarray as
    index and each data in a separate pickle file.

    Parameters
    ----------
    size : int
        The size of the storage.
    tag : str, optional
        Additional information to identify the storage.
    dir : str, optional
        The directory path for storage. If not provided, a default
        temporary directory is used.

    Attributes
    ----------
    directory : str
        The directory path of the storage.
    size : int
        The size of the storage.
    tag : str
        Additional information identifying the storage.
    fcache_dtype : dtype
        The dtype of the storage files.
    fcache_ : str
        The filename of the storage file.

    """

    def __init__(self, size, tag="", dir=None):
        if dir is None:
            suffix = "_" + tag
            dir = str(
                tempfile.mkdtemp(dir=_DEFAULT_STORAGE_DIR, suffix=suffix)
            )
        elif not os.path.isdir(dir) or os.listdir(dir):
            raise FileExistsError(f"'{dir!r}' must an empty directory")

        dtype = f"S{len(dir) + 200}"
        _, filename = tempfile.mkstemp(dir=dir, suffix="_fcache.mm")
        self._tag = tag
        self._files = np.memmap(filename, shape=size, dtype=dtype, mode="w+")

    @property
    def directory(self):
        """The directory path of the storage."""
        return os.path.dirname(self.fcache_)

    @property
    def size(self):
        """The size of the storage."""
        return len(self._files)

    @property
    def tag(self):
        """Additional information identifying the storage."""
        return self._tag

    @property
    def fcache_dtype(self):
        """The dtype of the storage files."""
        return self._files.dtype

    @property
    def fcache_(self):
        return self._files.filename

    @doc_inherit(StorageABC.lock)
    def lock(self):
        self._files.flush()
        self._files = np.memmap(
            self.fcache_, shape=self.size, dtype=self.fcache_dtype, mode="r"
        )

    def get(self, idx, default=None):
        """Get the data at index idx or the default value if it does not exist."""
        if idx >= len(self):
            raise IndexError(f"Index '{idx}' is out of range")
        filename = self._files[idx].decode()
        res = joblib.load(filename=filename) if filename else default
        return res

    @doc_inherit(StorageABC.__len__)
    def __len__(self):
        return self.size

    @doc_inherit(StorageABC.__setitem__)
    def __setitem__(self, idx, nddata):
        if idx > len(self):
            raise IndexError(f"Index '{idx}' is out of range")
        fd, filename = tempfile.mkstemp(
            dir=self.directory, suffix="_nddata.jpkl"
        )
        with open(fd, "wb") as fp:
            joblib.dump(nddata, fp)
        self._files[idx] = filename.encode()

    @doc_inherit(StorageABC.__setitem__)
    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            rargs = idxs.indices(len(self))
            idxs = range(*rargs)
        elif isinstance(idxs, (int, np.integer)):
            if idxs >= len(self):
                raise IndexError(f"Index '{idxs}' is out of range")
            return self.get(idxs)

        results = np.empty_like(idxs, dtype=object)
        for idx_to_insert, idx_to_search in enumerate(idxs):
            results[idx_to_insert] = self.get(idx_to_search)
        return results

    @doc_inherit(StorageABC.__repr__)
    def __repr__(self):
        directory = self.directory
        original_repr = super().__repr__()[1:-1]
        return f"<{original_repr} {directory!r}>"


# =============================================================================
# IN MEMORY Multi process
# =============================================================================


class MemoryStorage(StorageABC):
    """MemoryStorage class for storing data in memory.

    Upon initialization, the class creates an instance of DirectoryStorage
    for managing data accesed in multiprocess. When the lock() method is
    called, the class transitions to read-only mode and loads the entire
    dataset into memory for fast access.

    Parameters
    ----------
    size : int
        The size of the storage.
    tag : str, optional
        Additional information to identify the storage.
    dir : str, optional
        The directory path for storage before the lock() call.
        If not provided, a temporary directory is used.
    lock_clean : bool, optional
        If True, removes the directory when locking.

    """

    def __init__(self, size, tag="", dir=None, lock_clean=True):
        self._dir_storage = DirectoryStorage(size=size, tag=tag, dir=dir)
        self._data = None
        self._lock_clean = bool(lock_clean)

    @doc_inherit(StorageABC.lock)
    def lock(self):
        self._dir_storage.lock()
        self._data = np.array(self._dir_storage)
        self._data.setflags(write=False)
        if self._lock_clean:
            shutil.rmtree(self._dir_storage.directory)
            self._dir_storage = None

    @doc_inherit(StorageABC.__len__)
    def __len__(self):
        return (
            len(self._dir_storage) if self._data is None else len(self._data)
        )

    @doc_inherit(StorageABC.__getitem__)
    def __getitem__(self, idx):
        return self._data.__getitem__(idx)

    @doc_inherit(StorageABC.__setitem__)
    def __setitem__(self, idx, v):
        self._dir_storage.__setitem__(idx, v)


# =============================================================================
# MEMORY MONO PROCESS
# =============================================================================


class SingleProcessInMemoryStorage(StorageABC):
    """SingleProcessStorage class for storing data in a single process.

    Este storage solo almacena los datos en memoria sobre un numpy array
    pero implentando la interfaz StorageABC.

    Parameters
    ----------
    size : int
        The size of the storage.
    tag : str, optional
        Additional information to identify the storage.

    """

    def __init__(self, size, tag=""):
        self._data = np.full(size, None, dtype=object)

    @doc_inherit(StorageABC.lock)
    def lock(self):
        self._data.setflags(write=False)

    @doc_inherit(StorageABC.__len__)
    def __len__(self):
        return len(self._data)

    @doc_inherit(StorageABC.__getitem__)
    def __getitem__(self, idx):
        return self._data.__getitem__(idx)

    def __setitem__(self, idx, v):
        self._data.__setitem__(idx, v)


# =============================================================================
# FACTORY
# =============================================================================


_STORAGES = bidict.bidict(
    {
        "directory": DirectoryStorage,
        "memory": MemoryStorage,
        "single_process": SingleProcessInMemoryStorage,
    }
)


@contextlib.contextmanager
def storage(storage_type, size, tag, **kwargs):
    """Context manager for result storage.

    This context manager creates and manages a result storage instance based on
    the provided storage type. It yields the storage instance to the context
    block for use. After the context block is executed, it locks the storage
    instance.

    Parameters
    ----------
    storage_type : str
        The type of result storage to create.
    size : int
        The size of the storage.
    tag: str
        Some extra information to identify for "what" you need an storage.
    **kwargs
        Additional keyword arguments specific to the storage type.

    Yields
    ------
    result_storage_instance
        An instance of the result storage for use within the context block.

    Example
    -------
    >>> with storage("type1", 100) as stg:
    ...     stg[0] = "data1"

    """
    # load the class of the storage
    cls = _STORAGES[storage_type]

    # create the storage
    stg = cls(size, tag, **kwargs)

    try:
        yield stg  # yield the storage to the context block
    finally:
        stg.lock()  # lock the storage
