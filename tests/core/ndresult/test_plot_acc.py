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

from unittest import mock

from matplotlib import pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest

import seaborn as sns

from skneuromsi.core import constants
from skneuromsi.core.ndresult import plot_acc


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.plot
@check_figures_equal()
def test_ResultPlotter_line_positions(random_ndresult, fig_test, fig_ref):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    plotter = plot_acc.ResultPlotter(ndres)

    test_axes = fig_test.subplots(1, 3, sharey=True)
    plotter.line_positions(ax=test_axes)

    # EXPECTED
    ref_axes = fig_ref.subplots(1, 3, sharey=True).flatten()

    time = ndres.stats.dimmax()[constants.D_TIMES]

    ll, tl = np.sort(ndres.position_range)

    xa = ndres.to_xarray()

    x0 = xa.sel(positions_coordinates="x0", times=time).to_dataframe()
    x1 = xa.sel(positions_coordinates="x1", times=time).to_dataframe()
    x2 = xa.sel(positions_coordinates="x2", times=time).to_dataframe()

    sns.lineplot(
        ax=ref_axes[0],
        x=constants.D_POSITIONS,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x0,
        legend=False,
        alpha=0.75,
    )
    ticks = ref_axes[0].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[0].set_xticks(ticks)
    ref_axes[0].set_xticklabels(labels)
    ref_axes[0].set_title("x0")

    sns.lineplot(
        ax=ref_axes[1],
        x=constants.D_POSITIONS,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x1,
        legend=False,
        alpha=0.75,
    )
    ticks = ref_axes[1].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[1].set_xticks(ticks)
    ref_axes[1].set_xticklabels(labels)
    ref_axes[1].set_title("x1")

    sns.lineplot(
        ax=ref_axes[2],
        x=constants.D_POSITIONS,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x2,
        legend=True,
        alpha=0.75,
    )
    ticks = ref_axes[2].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[2].set_xticks(ticks)
    ref_axes[2].set_xticklabels(labels)
    ref_axes[2].set_title("x2")

    fig_ref.suptitle(f"{ndres.mname} - Time {time}")


@pytest.mark.plot
@check_figures_equal()
def test_ResultPlotter_line_positions_default_axes(
    random_ndresult, fig_test, fig_ref
):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    plotter = plot_acc.ResultPlotter(ndres)

    test_axes = fig_test.subplots(1, 3, sharey=True)
    mock_subplots = (fig_test, test_axes)
    with mock.patch("matplotlib.pyplot.subplots", return_value=mock_subplots):
        plotter.line_positions()

    # EXPECTED
    ref_axes = fig_ref.subplots(1, 3, sharey=True)

    size_x, size_y = fig_ref.get_size_inches()
    fig_ref.set_size_inches(size_x * 3, size_y)

    plotter.line_positions(ax=ref_axes)


@pytest.mark.plot
@check_figures_equal()
def test_ResultPlotter_line_times(random_ndresult, fig_test, fig_ref):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    plotter = plot_acc.ResultPlotter(ndres)

    test_axes = fig_test.subplots(1, 3, sharey=True)
    plotter.line_times(ax=test_axes)

    # EXPECTED
    ref_axes = fig_ref.subplots(1, 3, sharey=True).flatten()

    position = ndres.stats.dimmax()[constants.D_POSITIONS]

    ll, tl = np.sort(ndres.time_range)

    xa = ndres.to_xarray()

    x0 = xa.sel(positions_coordinates="x0", positions=position).to_dataframe()
    x1 = xa.sel(positions_coordinates="x1", positions=position).to_dataframe()
    x2 = xa.sel(positions_coordinates="x2", positions=position).to_dataframe()

    sns.lineplot(
        ax=ref_axes[0],
        x=constants.D_TIMES,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x0,
        legend=False,
        alpha=0.75,
    )
    ticks = ref_axes[0].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[0].set_xticks(ticks)
    ref_axes[0].set_xticklabels(labels)
    ref_axes[0].set_title("x0")

    sns.lineplot(
        ax=ref_axes[1],
        x=constants.D_TIMES,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x1,
        legend=False,
        alpha=0.75,
    )
    ticks = ref_axes[1].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[1].set_xticks(ticks)
    ref_axes[1].set_xticklabels(labels)
    ref_axes[1].set_title("x1")

    sns.lineplot(
        ax=ref_axes[2],
        x=constants.D_TIMES,
        y=constants.XA_NAME,
        hue=constants.D_MODES,
        data=x2,
        legend=True,
        alpha=0.75,
    )
    ticks = ref_axes[2].get_xticks()
    ticks_array = np.asarray(ticks, dtype=float)
    tmin, tmax = np.min(ticks_array), np.max(ticks_array)
    new_ticks = np.interp(ticks_array, (tmin, tmax), (ll, tl))
    labels = np.array([f"{t:.2f}" for t in new_ticks])
    ref_axes[2].set_xticks(ticks)
    ref_axes[2].set_xticklabels(labels)
    ref_axes[2].set_title("x2")

    fig_ref.suptitle(f"{ndres.mname} - Position {position}")


@pytest.mark.plot
@check_figures_equal()
def test_ResultPlotter_line_times_default_axes(
    random_ndresult, fig_test, fig_ref
):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    plotter = plot_acc.ResultPlotter(ndres)

    test_axes = fig_test.subplots(1, 3, sharey=True)
    mock_subplots = (fig_test, test_axes)
    with mock.patch("matplotlib.pyplot.subplots", return_value=mock_subplots):
        plotter.line_times()

    # EXPECTED
    ref_axes = fig_ref.subplots(1, 3, sharey=True)

    size_x, size_y = fig_ref.get_size_inches()
    fig_ref.set_size_inches(size_x * 3, size_y)

    plotter.line_times(ax=ref_axes)


def test_ResultPlotter_invalid_number_of_axes(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
    )
    plotter = plot_acc.ResultPlotter(ndres)

    with pytest.raises(ValueError):
        plotter.line_positions(ax=plt.gca())
    with pytest.raises(ValueError):
        plotter.line_times(ax=plt.gca())
