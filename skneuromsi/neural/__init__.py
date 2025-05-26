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

"""Model class for Neural Network models of multisensory integration.

This module contains the implementation of multisensory integration models
based on neural networks.


"""

# =============================================================================
# IMPORTS
# =============================================================================

from ._cuppini2014 import Cuppini2014
from ._cuppini2017 import Cuppini2017
from ._paredes2025 import Paredes2025

__all__ = [
    "Cuppini2014",
    "Cuppini2017",
    "Paredes2025",
]
