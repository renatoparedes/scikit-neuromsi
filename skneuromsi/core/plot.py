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

"""Plot helper for the Result object."""

# =============================================================================
# IMPORTS
# =============================================================================

import seaborn as sns

from ..utils import AccessorABC


# =============================================================================
# PLOTTER OBJECT
# =============================================================================
class ResultPlotter(AccessorABC):
    """Make plots of Result.

    Kind of plot to produce:

    - 'hist' : criteria histogram.
    - 'box' : criteria boxplot.
    - 'kde' : criteria Kernel Density Estimation plot.

    """

    _default_kind = "line"

    def __init__(self, result):
        self._result = result

    # PRIVATE =================================================================
    # This method are used "a lot" inside all the different plots, so we can
    # save some lines of code

    @property
    def _df(self):
        # proxy to access the dataframe with the data
        return self._result._df

    @property
    def _criteria_labels(self):
        # list with all the criteria + objectives
        dm = self._result
        labels = [
            f"{c} {o.to_string()}" for c, o in zip(dm.criteria, dm.objectives)
        ]
        return labels

    # HIST ====================================================================

    def hist(self, **kwargs):
        ax = sns.histplot(self._df, **kwargs)
        return ax

    # BOX =====================================================================

    def box(self, **kwargs):
        ax = sns.boxplot(data=self._df, **kwargs)
        return ax

    # LINE ====================================================================

    def line(self, **kwargs):
        ax = sns.lineplot(data=self._df, **kwargs)
        return ax
