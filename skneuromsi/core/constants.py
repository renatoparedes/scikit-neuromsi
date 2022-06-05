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

"""Constants used all skneuromsi core modules."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

D_MODES = "modes"
D_TIMES = "times"
D_POSITIONS = "positions"
D_POSITIONS_COORDINATES = "positions_coordinates"

DIMENSIONS = np.array([D_MODES, D_TIMES, D_POSITIONS, D_POSITIONS_COORDINATES])

XA_NAME = "values"
