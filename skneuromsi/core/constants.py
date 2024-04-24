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

#: Name of the "modes" dimension to be stored within the NDResult.
D_MODES = "modes"

#: Name of the "times" dimension to be stored within the NDResult.
D_TIMES = "times"

#: Name of the "positions" dimension to be stored within the NDResult.
D_POSITIONS = "positions"

#: Name of the "positions_coordinates" dimension within the NDResult.
D_POSITIONS_COORDINATES = "positions_coordinates"

#: Array containing all dimension names used in NDResult.
DIMENSIONS = np.array([D_MODES, D_TIMES, D_POSITIONS, D_POSITIONS_COORDINATES])

#: Name of the xarray.DataArray internal to the NDResult.
XA_NAME = "values"

#: Default data type for NDResult.
DEFAULT_FLOAT_DTYPE = np.float32
