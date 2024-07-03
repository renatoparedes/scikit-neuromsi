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

"""Core functionalities and structures of skneuromsi."""

# =============================================================================
# IMPORTS
# =============================================================================

from .modelabc import ParameterAliasTemplate, SKNMSIMethodABC, SKNMSIRunConfig
from .ndcollection import NDResultCollection
from .result import NDResult, compress_ndresult, decompress_ndresult

# =============================================================================
# ALL
# =============================================================================


__all__ = [
    "compress_ndresult",
    "decompress_ndresult",
    "ParameterAliasTemplate",
    "SKNMSIRunConfig",
    "SKNMSIMethodABC",
    "NDResult",
    "NDResultCollection",
]
