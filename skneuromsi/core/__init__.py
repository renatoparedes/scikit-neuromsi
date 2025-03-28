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

"""Core functionalities and structures of skneuromsi."""

# =============================================================================
# IMPORTS
# =============================================================================

from .modelabc import ParameterAliasTemplate, SKNMSIMethodABC, SKNMSIRunConfig
from .ndresult import (
    CompressedNDResult,
    DEFAULT_COMPRESSION_PARAMS,
    NDResult,
    compress_ndresult,
    decompress_ndresult,
    validate_compression_params,
)

# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = (0, 2, 0)

# =============================================================================
# ALL
# =============================================================================


__all__ = [
    "DEFAULT_COMPRESSION_PARAMS",
    "VERSION",
    "CompressedNDResult",
    "NDResult",
    "ParameterAliasTemplate",
    "SKNMSIMethodABC",
    "SKNMSIRunConfig",
    "compress_ndresult",
    "decompress_ndresult",
    "validate_compression_params",
]
