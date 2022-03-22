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

import functools

import numpy as np

import pandas as pd

from .plot import ResultPlotter
from .stats import ResultStatsAccessor

# =============================================================================
# CLASS RESULT
# =============================================================================


class Result:
    def __init__(self, name, data):
        self._name = name
        self._df = (
            data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        )

    @property
    def name(self):
        return self._name

    def __getattr__(self, a):
        if a in self._df.columns:
            return self._df[a].copy()
        raise AttributeError(a)

    def __getitem__(self, a):
        return self._df[a].copy()

    def __dir__(self):
        return super().__dir__() + list(self._df.columns)

    def __len__(self):
        return len(self._df)

    def copy(self):
        cls = type(self)
        return cls(self._name, self._df.copy())

    @property
    def shape(self):
        return np.shape(self._df)

    def _get_dimensions(self):
        p_number, m_number = self.shape
        dimensions = f"{p_number} Positions x {m_number} Modes"
        return dimensions

    def __repr__(self):
        """dm.__repr__() <==> repr(dm)."""

        dimensions = self._get_dimensions()

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        max_cols = pd.get_option("display.max_columns")
        max_colwidth = pd.get_option("display.max_colwidth")

        width = (
            pd.io.formats.console.get_console_size()[0]
            if pd.get_option("display.expand_frame_repr")
            else None
        )

        original_string = self._df.to_string(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            line_width=width,
            max_colwidth=max_colwidth,
            show_dimensions=False,
        )

        # add dimension
        string = f"{original_string}\n[{self._name} - {dimensions}]"

        return string

    def _repr_html_(self):
        """Return a html representation.

        Mainly for IPython notebook.
        """
        dimensions = self._get_dimensions()

        # retrieve the original string
        with pd.option_context("display.show_dimensions", False):
            original_html = self._df._repr_html_()

        # add dimension
        html = (
            "<div class='result'>\n"
            f"{original_html}"
            f"<em class='result-dim'>{self._name} - {dimensions}</em>\n"
            "</div>"
        )

        return html

    # ACCESSORS ===============================================================

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        """Plot accessor."""
        return ResultPlotter(self)

    @property
    @functools.lru_cache(maxsize=None)
    def stats(self):
        """Descriptive statistics accessor."""
        return ResultStatsAccessor(self)
