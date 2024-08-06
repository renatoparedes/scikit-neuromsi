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

"""Tests for the `skneuromsi.utils.readout_tools` module."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skneuromsi.utils import neural_tools, readout_tools

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_data_vector(n_peaks=3, peak_height=0.15, size=180):
    if n_peaks == 1:
        pos = size / 2
    elif n_peaks == 2:
        pos = size / 4
    else:
        pos = size / (n_peaks - 1)

    # Create an array of x values
    x = (
        neural_tools.calculate_stimuli_input(
            intensity=1, scale=4, loc=pos, neurons=int(size / n_peaks)
        )
        * peak_height
    )

    return np.tile(x, n_peaks)


# =============================================================================
# TEST TOOLS
# =============================================================================


@pytest.mark.parametrize(
    "peak_values, result", [([0.4, 0.2], 0.08), ([0.4, 0.2, 0.5], 0.105)]
)
def test_calculate_multiple_peaks_probability(peak_values, result):
    res = readout_tools.calculate_multiple_peaks_probability(peak_values)
    np.testing.assert_almost_equal(result, res)


@pytest.mark.parametrize(
    "peak_values, result",
    [
        ([], 0),
        ([0.42], 0.42),
        ([0.4, 0.2], 1 - 0.08),
        ([0.4, 0.2, 0.5], 1 - 0.105),
    ],
)
def test_calculate_single_peak_probability(peak_values, result):
    res = readout_tools.calculate_single_peak_probability(peak_values)
    np.testing.assert_almost_equal(result, res)


@pytest.mark.parametrize("causes_kind", [("count"), ("prob"), (None)])
def test_calculate_causes_from_peaks(causes_kind):
    rng = np.random.default_rng(seed=700)
    n_peaks = 3
    peak_height = rng.random(1)[0]
    mode_values = generate_data_vector(
        n_peaks=n_peaks, peak_height=peak_height
    )

    causes = readout_tools.calculate_causes_from_peaks(
        mode_values, causes_kind, peak_height - 0.10
    )
    estimated_n_peaks = n_peaks
    estimated_prob_values = np.repeat(peak_height, estimated_n_peaks)

    if causes_kind == "count":
        np.testing.assert_almost_equal(causes, estimated_n_peaks)
    elif causes_kind == "prob":
        causes_prob = readout_tools.calculate_single_peak_probability(
            estimated_prob_values
        )
        np.testing.assert_almost_equal(causes, causes_prob)


@pytest.mark.parametrize("causes_dim", [("space"), ("time"), (None)])
def test_calculate_spatiotemporal_causes_from_peaks(causes_dim):
    spatial_causes = 1
    spatial_mode_values = generate_data_vector(
        n_peaks=spatial_causes, peak_height=0.45, size=180
    )
    temporal_causes = 2
    spatiotemporal_mode_values = neural_tools.create_unimodal_stimuli_matrix(
        neurons=180,
        stimuli=spatial_mode_values,
        stimuli_duration=10,
        onset=20,
        simulation_length=400,
        time_res=0.01,
        dt=0.01,
        stimuli_n=temporal_causes,
        soa=100,
    )

    # Smooth signal
    x = np.arange(len(spatiotemporal_mode_values))
    sigma = 2000
    gaussian = np.exp(-(np.square(x)) / (2 * np.square(sigma))) / 1000
    result = np.convolve(
        spatiotemporal_mode_values[:, 90], gaussian, mode="full"
    )
    spatiotemporal_mode_values[:, 90] = result[:40000]

    if causes_dim == "space":
        estimated_spatial_causes = (
            readout_tools.calculate_spatiotemporal_causes_from_peaks(
                mode_spatiotemporal_activity_data=spatiotemporal_mode_values,
                causes_kind="count",
                causes_dim=causes_dim,
                time_point=3000,
                spatial_point=90,
                peak_threshold=0.15,
                lookahead=10,
            )
        )
        np.testing.assert_almost_equal(
            spatial_causes, estimated_spatial_causes
        )

    elif causes_dim == "time":
        estimated_temporal_causes = (
            readout_tools.calculate_spatiotemporal_causes_from_peaks(
                mode_spatiotemporal_activity_data=spatiotemporal_mode_values,
                causes_kind="count",
                causes_dim=causes_dim,
                time_point=3000,
                spatial_point=90,
                peak_threshold=0.15,
                lookahead=10,
            )
        )
        np.testing.assert_almost_equal(
            temporal_causes, estimated_temporal_causes
        )
