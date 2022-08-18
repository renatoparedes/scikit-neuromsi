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

import pytest

from skneuromsi.neural import Cuppini2014

# =============================================================================
# CUPPINI 2014
# =============================================================================


@pytest.mark.parametrize(
    "loc, intensity, scale", [(90, 10, 6), (70, 20, 7), (110, 30, 6)]
)
@pytest.mark.model
def test_cuppini2014_stim_generation(
    loc, intensity, scale
):  # TODO Include values far from centre. Fix scale evaluation: maybe fit gaussian.
    model = Cuppini2014()
    stim = model.stimuli_input(intensity=intensity, scale=scale, loc=loc)
    stim_loc = stim.argmax()
    stim_intensity = stim.max()
    stim_scale = stim.std()

    np.testing.assert_almost_equal(loc, stim_loc)
    np.testing.assert_almost_equal(intensity, stim_intensity)
    # np.testing.assert_almost_equal(scale, stim_scale)


@pytest.mark.parametrize(
    "loc, onset, duration", [(70, 16, 10), (90, 24, 10), (110, 8, 20)]
)
@pytest.mark.model
def test_cuppini2014_stim_matrix_generation_single(duration, loc, onset):
    model = Cuppini2014()
    stim = model.stimuli_input(intensity=30, scale=32, loc=loc)
    simulation_length = 250

    matrix = model.create_unimodal_stimuli_matrix(
        stimuli=stim,
        stimuli_duration=duration,
        onset=onset,
        simulation_length=simulation_length,
        stimuli_n=1,
    )

    onset_sim_time = int(onset * (1 / model._integrator.dt))
    duration_sim_time = int(duration * (1 / model._integrator.dt))

    space_res = matrix[onset_sim_time]
    time_res = matrix[:, loc]

    stim_matrix_loc = space_res.argmax()
    stim_matrix_onset = time_res.argmax()
    stim_matrix_duration = np.where(time_res == time_res.max())[0].size

    np.testing.assert_almost_equal(stim_matrix_loc, loc)
    np.testing.assert_almost_equal(stim_matrix_onset, onset_sim_time)
    np.testing.assert_almost_equal(stim_matrix_duration, duration_sim_time)


@pytest.mark.parametrize("soa, duration", [(90, 20), (70, 30), (110, 10)])
@pytest.mark.model
def test_cuppini2014_stim_matrix_generation_double(soa, duration):
    model = Cuppini2014()
    loc = 90
    onset = 16
    simulation_length = 250
    nstim = 2

    stim = model.stimuli_input(intensity=30, scale=32, loc=loc)

    matrix = model.create_unimodal_stimuli_matrix(
        stimuli=stim,
        stimuli_duration=duration,
        onset=onset,
        simulation_length=simulation_length,
        stimuli_n=nstim,
        soa=soa,
    )

    time_res = matrix[:, loc]
    soa_sim_time = int(soa * (1 / model._integrator.dt))
    duration_sim_time = int(duration * (1 / model._integrator.dt))

    stim_matrix_duration = np.where(time_res == time_res.max())[0].size / nstim

    tpoint = int(time_res.argmax() + duration * (1 / model._integrator.dt))
    stim_matrix_soa = matrix[tpoint:, loc].argmax()

    np.testing.assert_almost_equal(stim_matrix_duration, duration_sim_time)
    np.testing.assert_almost_equal(stim_matrix_soa, soa_sim_time)


@pytest.mark.parametrize("weight", [(0.50), (0.25), (0.75)])
@pytest.mark.model
def test_cuppini2014_synapses(weight):
    model = Cuppini2014()
    synapses = model.synapses(weight=weight)

    np.testing.assert_almost_equal(np.unique(synapses), weight)


@pytest.mark.parametrize(
    "excitation_loc, inhibition_loc", [(90, 20), (30, 70), (110, 50)]
)
@pytest.mark.model
def test_cuppini2014_lateral_synapses(excitation_loc, inhibition_loc):
    # Fix scale evaluation: maybe fit gaussian.
    model = Cuppini2014()
    exc = np.zeros((180, 180))
    inh = np.zeros((180, 180))

    for neuron_i in range(180):
        for neuron_j in range(180):
            distance = model.distance(neuron_i, neuron_j)
            exc[neuron_i, neuron_j] = model.lateral_synapse(distance, 2, 3)
            inh[neuron_i, neuron_j] = model.lateral_synapse(distance, 2, 24)

    exc_loc = exc[:, excitation_loc].argmax()
    inh_loc = inh[:, inhibition_loc].argmax()

    np.testing.assert_almost_equal(exc_loc, excitation_loc)
    np.testing.assert_almost_equal(inh_loc, inhibition_loc)


## TODO Synapses and Temporal Filter. Then compare with paper temporal evolution.
## TODO Include peak readout from paper.
