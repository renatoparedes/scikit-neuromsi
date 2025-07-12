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

"""Model class for Bayesian models of multisensory integration.

This module contains the implementation of multisensory integration models
based on Bayesian inference.


"""

# =============================================================================
# IMPORTS
# =============================================================================

from ._kording2007 import Kording2007
from ._zhu2024 import Zhu2024

__all__ = [
    "Kording2007",
    "Zhu2024",
]
