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

from skneuromsi.cuppini2014 import Cuppini2014

# =============================================================================
# CUPPINI 2014
# =============================================================================


@pytest.mark.parametrize(
    "loc, stim_location", [(90, 90), (70, 70), (110, 110)]
)
@pytest.mark.model
def test_cuppini2014_stim_generation(
    loc, stim_location
):  # TODO Include values far from centre.
    model = Cuppini2014()
    stim = model.stimuli_input(intensity=30, scale=32, loc=loc)
    stim_loc = stim.argmax()

    np.testing.assert_almost_equal(stim_location, stim_loc)


@pytest.mark.parametrize("loc, onset, duration", [(70, 16, 10), (90, 14, 10)])
@pytest.mark.model
def test_cuppini2014_stim_matrix_generation_single(
    duration, loc, onset
):  # TODO include more examples.
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


@pytest.mark.parametrize("soa, duration", [(90, 90), (70, 56), (110, 110)])
@pytest.mark.model
def test_cuppini2014_stim_matrix_generation_double(
    soa, duration
):  # TODO complete test
    model = Cuppini2014()
    loc = 90
    stim = model.stimuli_input(intensity=30, scale=32, loc=loc)

    duration = 15
    onset = 16
    simulation_length = 250

    matrix = model.create_unimodal_stimuli_matrix(
        stimuli=stim,
        stimuli_duration=duration,
        onset=onset,
        simulation_length=simulation_length,
        stimuli_n=2,
        soa=soa,
    )

    #   np.testing.assert_almost_equal(stim_location, stim_loc)

    assert True
