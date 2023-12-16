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

from findpeaks import findpeaks

# =============================================================================
# CUPPINI 2014
# =============================================================================


@pytest.mark.parametrize(
    "loc, intensity, scale", [(90, 10, 6), (70, 20, 7), (110, 30, 6)]
)
@pytest.mark.model
def test_cuppini2014_stim_generation(loc, intensity, scale):
    # TODO: Include values far from centre.
    # TODO: Fix scale evaluation: maybe fit gaussian.
    model = Cuppini2014()
    stim = model.stimuli_input(intensity=intensity, scale=scale, loc=loc)
    stim_loc = stim.argmax()
    stim_intensity = stim.max()
    # stim_scale = stim.std()

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
    model = Cuppini2014(time_range=(0, 250))
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


@pytest.mark.parametrize(
    "excitation_position, inhibition_position", [(90, 20), (30, 70), (110, 50)]
)
@pytest.mark.model
def test_cuppini2014_lateral_synapse(excitation_position, inhibition_position):
    # Fix scale evaluation: maybe fit gaussian.
    model = Cuppini2014()
    exc = np.zeros((180, 180))
    inh = np.zeros((180, 180))

    for neuron_i in range(180):
        for neuron_j in range(180):
            distance = model.distance(neuron_i, neuron_j)
            exc[neuron_i, neuron_j] = model.lateral_synapse(distance, 2, 3)
            inh[neuron_i, neuron_j] = model.lateral_synapse(distance, 2, 24)

    exc_pos = exc[:, excitation_position].argmax()
    inh_pos = inh[:, inhibition_position].argmax()

    np.testing.assert_almost_equal(exc_pos, excitation_position)
    np.testing.assert_almost_equal(inh_pos, inhibition_position)


@pytest.mark.parametrize("lateral_location", [(90), (30), (110), (50)])
@pytest.mark.model
def test_cuppini2014_lateral_synapses(lateral_location):
    model = Cuppini2014()
    latsynapses = model.lateral_synapses(
        excitation_loc=5,
        inhibition_loc=2,
        excitation_scale=3,
        inhibition_scale=24,
    )

    lat_self = latsynapses[:, lateral_location][lateral_location]

    np.testing.assert_almost_equal(lat_self, 0)


def test_cuppini2014_temporal_filter_auditory_stimuli():
    # In Raij 2010, for a single auditory stimuli
    #  Hechsl Gyrus auditory response latency is 23 ms and calcarine cortex visual response latency is 53 ms.

    model = Cuppini2014()
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=1,
        auditory_stim_n=1,
        visual_stim_n=0,
        auditory_duration=15,
        visual_duration=20,
        onset=25,
    )
    visual_latency = (np.argmax(res.e_.visual_total_input[:, 90]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 90]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 53, atol=18)
    np.testing.assert_allclose(auditory_latency, 23, atol=1)


def test_cuppini2014_temporal_filter_visual_stimuli():
    # In Raij 2010, for a single visual stimuli
    # Calcarine cortex visual response latency is 43 ms and Hechsl Gyrus auditory response latency is 82 ms.

    model = Cuppini2014(time_range=(0, 125))
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=1.05,
        auditory_stim_n=0,
        visual_stim_n=1,
        auditory_duration=15,
        visual_duration=20,
        onset=25,
    )

    visual_latency = (np.argmax(res.e_.visual_total_input[:, 90]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 90]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 43, atol=7)
    np.testing.assert_allclose(auditory_latency, 82, atol=3)


def test_cuppini2014_temporal_filter_audiovisual_stimuli():
    # In Raij 2010, for an audiovisual stimuli
    # Calcarine cortex visual response latency is 47 ms and Hechsl Gyrus auditory response latency is 23 ms.

    model = Cuppini2014(time_range=(0, 125))
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=0.9,
        auditory_stim_n=1,
        visual_stim_n=1,
        auditory_duration=15,
        visual_duration=20,
        onset=25,
    )
    visual_latency = (np.argmax(res.e_.visual_total_input[:, 90]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 90]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 47, atol=12)
    np.testing.assert_allclose(auditory_latency, 23, atol=1)


def test_cuppini2014_fission_illusion():
    model = Cuppini2014(time_range=(0, 250))
    res = model.run(
        auditory_intensity=2.30,
        visual_intensity=0.85,
        auditory_stim_n=2,
        visual_stim_n=1,
        auditory_duration=10,
        visual_duration=20,
        onset=25,
        soa=56,
    )
    fp = findpeaks(method="topology", verbose=0, limit=0.40)
    X = res.get_modes("visual").query("positions==90").visual.values
    results = fp.fit(X)
    flashes = results["df"].peak.sum()

    np.testing.assert_equal(flashes, 2)
