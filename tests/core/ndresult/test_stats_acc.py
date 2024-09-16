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

import numpy as np

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
    print(stats.min())
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
    print(stats.dimmin())
    np.testing.assert_allclose(stats.dimmin(), 0.004379153251647949)
