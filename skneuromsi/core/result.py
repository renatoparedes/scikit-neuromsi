#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses
import pandas as pd


@dataclasses.dataclass
class Result:
    _df: pd.DataFrame

    @classmethod
    def from_dict(self, columns):
        df = pd.DataFrame(columns)
        return type(self)(df)
