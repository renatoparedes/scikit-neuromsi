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

"""Implementation of multisensory integration models."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import testing
from .core import (
    CompressedNDResult,
    NAME,
    NDResult,
    VERSION,
    compress_ndresult,
    decompress_ndresult,
)
from .io import (
    open_ndresult,
    open_ndresults_collection,
    read_ndc,
    read_ndr,
    store_ndresult,
    store_ndresults_collection,
    to_ndc,
    to_ndr,
)
from .ndcollection import NDResultCollection


# =============================================================================
# META
# =============================================================================

__version__ = ".".join(map(str, VERSION))


__all__ = [
    "compress_ndresult",
    "decompress_ndresult",
    "CompressedNDResult",
    "NAME",
    "NDResult",
    "NDResultCollection",
    "read_ndr",
    "open_ndresult",
    "read_ndc",
    "open_ndresults_collection",
    "to_ndr",
    "store_ndresult",
    "to_ndc",
    "store_ndresults_collection",
    "testing",
]
