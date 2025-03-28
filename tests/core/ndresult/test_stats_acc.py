#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from skneuromsi.core.ndresult import stats_acc


# =============================================================================
# TESTS
# =============================================================================


def test_ResultStatsAccessor_count(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)
    assert stats.count() == 1200


def test_ResultStatsAccessor_mean(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)
    np.testing.assert_allclose(stats.mean(), 0.5046552419662476)


def test_ResultStatsAccessor_std(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)
    np.testing.assert_allclose(stats.std(), 0.2873277962207794)


def test_ResultStatsAccessor_min(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)
    np.testing.assert_allclose(stats.min(), 0.004379153251647949)


def test_ResultStatsAccessor_dimmin(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)

    expected = pd.Series(
        {
            "modes": "output",
            "times": 4,
            "positions": 18,
            "positions_coordinates": "x1",
            "values": 0.004379153251647949,
        },
        name="min",
    )

    pd.testing.assert_series_equal(stats.dimmin(), expected)


def test_ResultStatsAccessor_max(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)
    np.testing.assert_allclose(stats.max(), 0.9991046786308289)


def test_ResultStatsAccessor_dimmax(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)

    expected = pd.Series(
        {
            "modes": "output",
            "times": 0,
            "positions": 17,
            "positions_coordinates": "x1",
            "values": 0.9991046786308289,
        },
        name="max",
    )

    pd.testing.assert_series_equal(stats.dimmax(), expected)


def test_ResultStatsAccessor_quantile(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)

    np.testing.assert_allclose(
        stats.quantile(q=(0.25, 0.5, 0.75)),
        [0.25559422, 0.50596806, 0.75676683],
    )


def test_ResultStatsAccessor_describe(random_ndresult):
    ndres = random_ndresult(
        input_modes_min=1,
        input_modes_max=1,
        time_res=1,
        position_res=1,
        position_coordinates_min=3,
        position_coordinates_max=3,
        seed=42,
    )
    stats = stats_acc.ResultStatsAccessor(ndres)

    expected = pd.DataFrame(
        [
            1200.0,
            0.5046552419662476,
            0.2873277962207794,
            0.004379153251647949,
            0.2555942237377167,
            0.5059680640697479,
            0.7567668259143829,
            0.9991046786308289,
        ],
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        columns=["describe"],
    )

    pd.testing.assert_frame_equal(stats.describe(), expected)
