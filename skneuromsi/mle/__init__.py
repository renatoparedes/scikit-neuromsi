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

"""Model class for Maximum Likelihood models of multisensory integration.

This module contains the implementation of multisensory integration models
based on Maximum Likelihood Estimation.


"""

# =============================================================================
# IMPORTS
# =============================================================================

from ._alais_burr2004 import AlaisBurr2004

__all__ = [
    "AlaisBurr2004",
]
