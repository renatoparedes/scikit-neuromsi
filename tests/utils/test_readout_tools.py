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


def generate_data_vector(n_peaks=3, peak_height=0.15):
    size = 180
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
    num_peaks = np.array([3])
    peak_height = rng.random(1)[0]

    # Create an array of x values
    x = np.linspace(0, 2 * np.pi * num_peaks, 1000)

    # Generate sinusoidal wave
    y = np.sin(x).T[0]

    # Normalize the wave to be between 0 and 1
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y)) * peak_height

    causes = readout_tools.calculate_causes_from_peaks(
        y_normalized, causes_kind, peak_height - 0.05
    )
    estimated_n_peaks = num_peaks[0] - 1
    estimated_prob_values = np.repeat(peak_height, estimated_n_peaks)

    if causes_kind == "count":
        np.testing.assert_almost_equal(causes, estimated_n_peaks)
    elif causes_kind == "prob":
        causes_prob = readout_tools.calculate_single_peak_probability(
            estimated_prob_values
        )
        np.testing.assert_almost_equal(causes, causes_prob)
