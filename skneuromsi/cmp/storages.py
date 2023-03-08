import abc
import atexit
import contextlib
import shutil
import tempfile
from collections.abc import Sequence

import joblib


# =============================================================================
# ABC CLASS
# =============================================================================
class ResultsStorageABC(Sequence):
    @abc.abstractmethod
    def add(self, nddata):
        ...

    @abc.abstractmethod
    def lock(self):
        ...


# =============================================================================
# NO STORAGE
# =============================================================================
class NoStorage(ResultsStorageABC):
    def __init__(self, conf):
        ...

    def add(self, idx):
        ...

    def lock(self):
        ...

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(f"{type(self)} never has elements")


# =============================================================================
# IN MEMORY
# =============================================================================


class InMemoryResultsStorage(ResultsStorageABC):
    def __init__(self, conf):
        self._nddatas_lst = []

    def add(self, nddata):
        self._nddatas_lst.append(nddata)

    def lock(self):
        self._nddatas = tuple(self._nddatas_lst)
        del self._nddatas_lst

    def __len__(self):
        return len(self._nddatas)

    def __getitem__(self, idx):
        nddata = self._nddatas[idx]
        return nddata


# =============================================================================
# IN DISK
# =============================================================================


class InDiskResultsStorage(ResultsStorageABC):
    def __init__(self, conf):
        self._dir = tempfile.mkdtemp(suffix="_skneuromsi")
        self._idx_to_filename = []
        atexit.register(shutil.rmtree, self._dir)

    def lock(self):
        self._idx_to_filename = tuple(self._idx_to_filename)

    def add(self, nddata):
        fd, filename = tempfile.mkstemp(dir=self._dir, suffix="_nddata")
        self._idx_to_filename.append(filename)
        with open(fd, "wb") as fp:
            joblib.dump(nddata, fp)

    def __len__(self):
        return len(self._idx_to_filename)

    def __getitem__(self, idx):
        filename = self._idx_to_filename[idx]
        return joblib.load(filename=filename)


# =============================================================================
# FACTORY
# =============================================================================

_STORAGES = {
    None: NoStorage,
    "memory": InMemoryResultsStorage,
    "disk": InDiskResultsStorage,
}


@contextlib.contextmanager
def make_storage(result_storage_type):
    cls = _STORAGES[result_storage_type]
    stg = cls(result_storage_type)
    try:
        yield stg
    finally:
        stg.lock()
