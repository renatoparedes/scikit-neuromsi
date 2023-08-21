import abc
import atexit
import contextlib
import shutil
import tempfile
import os

import numpy as np

import joblib


# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_STORAGE_DIR = tempfile.mkdtemp(prefix="skneuromsi_")
atexit.register(shutil.rmtree, _DEFAULT_STORAGE_DIR)


# =============================================================================
# ABC CLASS
# =============================================================================
class StorageABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def lock(self):
        ...

    @abc.abstractmethod
    def __getitem__(self, idx):
        ...

    @abc.abstractmethod
    def __setitem__(self, idx, v):
        ...

    @abc.abstractmethod
    def __len__(self):
        ...

    def __delitem__(self, idx):
        del self[idx]

    def __repr__(self):
        stype = type(self).__name__
        filled = " ".join([("X" if e else "-") for e in self])
        repr_str = f"<{stype} [ {filled} ]>"
        return repr_str


# =============================================================================
# IN DISK
# =============================================================================


class DirectoryStorage(StorageABC):
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
        self._files = np.memmap(filename, shape=size, dtype=dtype, mode="w+")

    @property
    def directory(self):
        return os.path.dirname(self.fcache_)

    @property
    def size(self):
        return len(self._files)

    @property
    def fcache_dtype(self):
        return self._files.dtype

    @property
    def fcache_(self):
        return self._files.filename

    def lock(self):
        self._files.flush()
        self._files = np.memmap(
            self.fcache_, shape=self.size, dtype=self.fcache_dtype, mode="r"
        )

    def get(self, idx, default=None):
        filename = self._files[idx].decode()
        res = joblib.load(filename=filename) if filename else default
        return res

    def __len__(self):
        return self.size

    def __setitem__(self, idx, nddata):
        if idx > len(self):
            raise IndexError(f"Index '{idx}' is out of range")
        fd, filename = tempfile.mkstemp(
            dir=self.directory, suffix="_nddata.jpkl"
        )
        with open(fd, "wb") as fp:
            joblib.dump(nddata, fp)
        self._files[idx] = filename.encode()

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

    def __repr__(self):
        directory = self.directory
        original_repr = super().__repr__()[1:-1]
        return f"<{original_repr} {directory!r}>"


# =============================================================================
# IN MEMORY
# =============================================================================

class MemoryStorage(StorageABC):

    def __init__(self, size, tag="", dir=None, lock_clean=True):
        self._dir_storage = DirectoryStorage(size=size, tag=tag, dir=dir)
        self._data = None
        self._lock_clean = bool(lock_clean)

    def lock(self):
        self._dir_storage.lock()
        self._data = np.array(self._dir_storage)
        self._data.setflags(write=False)
        if self._lock_clean:
            shutil.rmtree(self._dir_storage.directory)
            self._dir_storage = None

    def __len__(self):
        return len(self._dir_storage) if self._data is None else len(self._data)

    def __getitem__(self, idx):
        return self._data.__getitem__(idx)

    def __setitem__(self, idx, v):
        self._dir_storage.__setitem__(idx, v)

# =============================================================================
# FACTORY
# =============================================================================


_STORAGES = {
    "directory": DirectoryStorage,
    "memory": MemoryStorage
}


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
    ...     stg.store("data1")
    ...
    """
    cls = _STORAGES[storage_type]
    stg = cls(size, tag, **kwargs)
    try:
        yield stg
    finally:
        stg.lock()
