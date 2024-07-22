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

"""Result class for multi-sensory integration simulation.

This module contains the implementation of the NDResult class which represents
the results of a multi-sensory integration simulation.


"""

# =============================================================================
# IMPORTS
# =============================================================================

from .compress import (
    DEFAULT_COMPRESSION_PARAMS,
    CompressedNDResult,
    compress_ndresult,
    decompress_ndresult,
    validate_compression_params,
)
from .result import NDResult

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "DEFAULT_COMPRESSION_PARAMS",
    "NDResult",
    "CompressedNDResult",
    "compress_ndresult",
    "decompress_ndresult",
    "validate_compression_params",
]
