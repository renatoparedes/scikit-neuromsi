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

""""""

# =============================================================================
# IMPORTS
# =============================================================================

from ..core import NDResult

# =============================================================================
# CONSTANTS
# =============================================================================


# =============================================================================
# RESULT COLLECTION
# =============================================================================


class NDResultCollection:
    def __init__(self, ndresults, name=None):
        self._name = str(name)
        self._ndresults = tuple(ndresults)

        if not all(isinstance(e, NDResult) for e in self._ndresults):
            raise TypeError(
                "All elements of 'ndresults' must be instances of NDResult"
            )

        import ipdb; ipdb.set_trace()


    def __repr__(self):
        cls_name = type(self).__name__
        name = self._name
        length = len(self._ndresults)
        return f"<{cls_name} name={name!r}, len={length}>"