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

"""Tests for the `skneuromsi.utils.neural_tools` module."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from scipy.optimize import curve_fit

from skneuromsi.utils import neural_tools


# =============================================================================
#  TEST ARCHITECTURE TOOLS
# =============================================================================


@pytest.mark.parametrize("lateral_location", [(90), (30), (110), (50)])
def test_calculate_lateral_synapses_auto_excitation(lateral_location):
    latsynapses = neural_tools.calculate_lateral_synapses(
        excitation_loc=5,
        inhibition_loc=2,
        excitation_scale=3,
        inhibition_scale=24,
        neurons=180,
    )

    lat_self = latsynapses[:, lateral_location][lateral_location]

    np.testing.assert_almost_equal(lat_self, 0)


@pytest.mark.parametrize(
    "excitation_position, inhibition_position", [(90, 20), (30, 70), (110, 50)]
)
def test_calculate_lateral_synapses_location(
    excitation_position, inhibition_position
):
    exc = np.zeros((180, 180))
    inh = np.zeros((180, 180))

    for neuron_i in range(180):
        for neuron_j in range(180):
            distance = neural_tools.calculate_neural_distance(
                neurons=180, position_j=neuron_i, position_k=neuron_j
            )
            exc[neuron_i, neuron_j] = 2 * np.exp(
                -(np.square(distance)) / (2 * np.square(3))
            )

            inh[neuron_i, neuron_j] = 2 * np.exp(
                -(np.square(distance)) / (2 * np.square(24))
            )

    exc_pos = exc[:, excitation_position].argmax()
    inh_pos = inh[:, inhibition_position].argmax()

    np.testing.assert_almost_equal(exc_pos, excitation_position)
    np.testing.assert_almost_equal(inh_pos, inhibition_position)


# =============================================================================
#  TEST STIMULI TOOLS
# =============================================================================


def gaussian(x, loc, scale, intensity):
    return intensity * np.exp(-(np.square(x - loc)) / (2 * np.square(scale)))


@pytest.mark.parametrize(
    "loc, intensity, scale", [(90, 10, 6), (70, 20, 7), (110, 30, 6)]
)
def test_calculate_stimuli_input(loc, intensity, scale):
    stim = neural_tools.calculate_stimuli_input(
        intensity=intensity, scale=scale, loc=loc, neurons=180
    )

    # Create an array representing the x-axis
    x = np.arange(len(stim))

    # Find the peak value and its position
    peak_idx = np.argmax(stim)
    peak_value = stim[peak_idx]

    # Estimate initial parameters
    initial_guess = [peak_idx, 1.0, peak_value]

    # Fit the Gaussian function to the data
    params, _ = curve_fit(gaussian, x, stim, p0=initial_guess)

    # Extract the mean and standard deviation from the fit parameters
    stim_loc, stim_scale, stim_intensity = params

    np.testing.assert_almost_equal(loc, stim_loc, decimal=3)
    np.testing.assert_almost_equal(scale, stim_scale, decimal=3)
    np.testing.assert_almost_equal(intensity, stim_intensity, decimal=3)


@pytest.mark.parametrize(
    "loc, onset, duration", [(70, 16, 10), (90, 24, 10), (110, 8, 20)]
)
def test_create_unimodal_stimuli_matrix(duration, loc, onset):
    # Create stimuli matrix with a single stimulus
    stim = neural_tools.calculate_stimuli_input(
        intensity=30, scale=32, loc=loc, neurons=180
    )
    simulation_length = 250

    matrix = neural_tools.create_unimodal_stimuli_matrix(
        stimuli=stim,
        stimuli_duration=duration,
        onset=onset,
        simulation_length=simulation_length,
        stimuli_n=1,
        neurons=180,
        dt=0.01,
        time_res=0.01,
    )

    # Define the expected onset, duration and location of stimulus
    onset_sim_time = int(onset * (1 / 0.01))
    duration_sim_time = int(duration * (1 / 0.01))

    space_res = matrix[onset_sim_time]
    time_res = matrix[:, loc]

    stim_matrix_loc = space_res.argmax()
    stim_matrix_onset = time_res.argmax()
    stim_matrix_duration = np.where(time_res == time_res.max())[0].size

    # Compare the expected values with the ones observed in the generated stimuli matrix
    np.testing.assert_almost_equal(stim_matrix_loc, loc)
    np.testing.assert_almost_equal(stim_matrix_onset, onset_sim_time)
    np.testing.assert_almost_equal(stim_matrix_duration, duration_sim_time)


@pytest.mark.parametrize(
    "soa, duration, nstim", [(90, 20, 2), (70, 30, 2), (110, 10, 2)]
)
def test_create_unimodal_multiple_stimuli_matrix(soa, duration, nstim):
    loc = 90
    onset = 16
    simulation_length = 250

    # Create stimuli matrix with multiple stimuli
    stim = neural_tools.calculate_stimuli_input(
        intensity=30, scale=32, loc=loc, neurons=180
    )

    matrix = neural_tools.create_unimodal_stimuli_matrix(
        stimuli=stim,
        stimuli_duration=duration,
        onset=onset,
        simulation_length=simulation_length,
        stimuli_n=nstim,
        soa=soa,
        neurons=180,
        dt=0.01,
        time_res=0.01,
    )

    time_res = matrix[:, loc]
    soa_sim_time = int(soa * (1 / 0.01))
    duration_sim_time = int(duration * (1 / 0.01))

    stim_matrix_duration = np.where(time_res == time_res.max())[0].size / nstim

    tpoint = int(time_res.argmax() + duration * (1 / 0.01))
    stim_matrix_soa = matrix[tpoint:, loc].argmax()

    np.testing.assert_almost_equal(stim_matrix_duration, duration_sim_time)
    # np.testing.assert_almost_equal(stim_matrix_soa, soa_sim_time) TODO Fix
