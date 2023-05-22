import abc
import atexit
import contextlib
import shutil
import tempfile
import os

import numpy as np

import joblib


# =============================================================================
# COERCION
# =============================================================================

_DEFAULT_STORAGE_DIR = tempfile.mkdtemp(prefix="skneuromsi_")
atexit.register(shutil.rmtree, _DEFAULT_STORAGE_DIR)


def coerce_storage_path(candidate, suffix):
    if candidate is None:
        candidate = tempfile.mkdtemp(dir=_DEFAULT_STORAGE_DIR, suffix=suffix)
    return str(candidate)


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
        repr_list = [("X" if e else "-") for e in self]
        repr_str = "[" + " ".join(repr_list) + "]"
        return repr_str


# =============================================================================
# IN DISK
# =============================================================================


class DirectoryStorage(StorageABC):
    def __init__(self, size, dir):
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
        cls_name = type(self).__name__
        directory = self.directory
        original_repr = super().__repr__()
        return f"<{cls_name} {directory!r} {original_repr}>"


# =============================================================================
# FACTORY
# =============================================================================


_STORAGES = {
    "directory": DirectoryStorage,
}


@contextlib.contextmanager
def storage(result_storage_type, size, **kwargs):
    cls = _STORAGES[result_storage_type]
    stg = cls(size, **kwargs)
    try:
        yield stg
    finally:
        stg.lock()