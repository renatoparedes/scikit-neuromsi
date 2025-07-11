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

"""test for skneuromsi.ndcollection.cplot_acc"""

# =============================================================================
# IMPORTS
# =============================================================================

from matplotlib.testing.decorators import check_figures_equal

import pytest

import seaborn as sns

from skneuromsi.ndcollection import cplot_acc

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal()
@pytest.mark.parametrize("n", [0, 1])
def test_NDResultCollectionPlotter_nreport(
    random_ndcollection, fig_test, fig_ref, n
):
    coll = random_ndcollection(
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
    y = the_report["Causes"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="Causes")

    ref_ax.set_ylabel(f"Proportion of {n} causes")


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal()
def test_NDResultCollectionPlotter_unity_report(
    random_ndcollection, fig_test, fig_ref
):
    coll = random_ndcollection(
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
    y = the_report["Causes"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="Causes")

    ref_ax.set_ylabel("Proportion of unique causes")


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal()
def test_NDResultCollectionPlotter_mean_report(
    random_ndcollection, fig_test, fig_ref
):
    coll = random_ndcollection(
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
    y = the_report["Causes"]
    sns.lineplot(x=x, y=y, ax=ref_ax, label="Causes")

    ref_ax.set_ylabel("Mean of causes")


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal()
def test_NDResultCollectionPlotter_bias(
    random_ndcollection, fig_test, fig_ref
):
    coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    plotter = cplot_acc.NDResultCollectionPlotter(coll)

    test_ax = fig_test.subplots()
    plotter.bias(
        influence_parameter="p0",
        changing_parameter="p1",
        ax=test_ax,
        quiet=True,
    )

    # EXPECTED
    ref_ax = fig_ref.subplots()

    the_bias = coll.bias.bias(
        "p0",
        changing_parameter="p1",
        quiet=True,
    )

    x = the_bias.index
    for column in the_bias.columns:
        y = the_bias[column]
        sns.lineplot(
            x=x,
            y=y,
            ax=ref_ax,
            linestyle="--",
            alpha=0.25,
            legend=False,
            label="Iterations" if column == the_bias.columns[0] else None,
        )

    the_bias_mean = coll.bias.bias_mean(
        "p0",
        changing_parameter="p1",
        quiet=True,
    )
    x_mean = the_bias_mean.index
    y_mean = the_bias_mean[the_bias_mean.columns[0]]
    sns.lineplot(
        x=x_mean,
        y=y_mean,
        ax=ref_ax,
        legend=False,
        label="Bias Mean",
        color="black",
    )

    ref_ax.set_ylabel("Bias")
    ref_ax.set_title(
        f"Fixed=p0, Target=p1\n"
        f"Mode={coll.coerce_mode(None)}, "
        f"Dimension={coll.coerce_dimension(None)}"
    )

    ref_ax.legend()
    leg = ref_ax.get_legend()
    leg.legend_handles[0].set_color("black")


@pytest.mark.plot
def test_NDResultCollectionPlotter_bias_show_iterations_and_mean_False(
    random_ndcollection,
):
    coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    plotter = cplot_acc.NDResultCollectionPlotter(coll)

    error = (
        "If 'show_iterations' and 'show_mean' are False, "
        "yout plot are empty"
    )
    with pytest.raises(ValueError, match=error):
        plotter.bias(
            influence_parameter="p0", show_iterations=False, show_mean=False
        )
