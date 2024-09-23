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

"""test for skneuromsi.ndcollection.cplot_acc

"""

# =============================================================================
# IMPORTS
# =============================================================================

import io
from unittest import mock

from matplotlib import pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import seaborn as sns


import skneuromsi as sknmsi
from skneuromsi.ndcollection import bias_acc, causes_acc, collection, cplot_acc

# =============================================================================
# TESTS
# =============================================================================


@check_figures_equal()
@pytest.mark.parametrize("n", [0, 1])
def test_NDResultCollectionPlotter_nreport(
    random_ndcollection, fig_test, fig_ref, n
):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    plotter = cplot_acc.NDResultCollectionPlotter(coll)

    test_ax = fig_test.subplots()
    plotter.n_report(n=n, parameter="p1", ax=test_ax)

    # EXPECTED
    ref_ax = fig_ref.subplots()

    the_report = coll.causes.n_report(n=n, parameter="p1")

    x = the_report.index
    y = the_report["p1"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="p1")

    ref_ax.set_ylabel(f"Proportion of {n} causes")


@check_figures_equal()
def test_NDResultCollectionPlotter_unity_report(
    random_ndcollection, fig_test, fig_ref
):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    plotter = cplot_acc.NDResultCollectionPlotter(coll)

    test_ax = fig_test.subplots()
    plotter.unity_report(parameter="p1", ax=test_ax)

    # EXPECTED
    ref_ax = fig_ref.subplots()

    the_report = coll.causes.unity_report(parameter="p1")

    x = the_report.index
    y = the_report["p1"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="p1")

    ref_ax.set_ylabel("Proportion of unit causes")


@check_figures_equal()
def test_NDResultCollectionPlotter_mean_report(
    random_ndcollection, fig_test, fig_ref
):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    plotter = cplot_acc.NDResultCollectionPlotter(coll)

    test_ax = fig_test.subplots()
    plotter.mean_report(parameter="p1", ax=test_ax)

    # EXPECTED
    ref_ax = fig_ref.subplots()

    the_report = coll.causes.mean_report(parameter="p1")

    x = the_report.index
    y = the_report["p1"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="p1")

    ref_ax.set_ylabel("Mean of causes")