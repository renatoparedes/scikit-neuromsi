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

from skneuromsi.neural import Paredes2022

from findpeaks import findpeaks

# =============================================================================
# PAREDES 2022
# =============================================================================


@pytest.mark.slow
@pytest.mark.model
def test_paredes2022_temporal_filter_auditory_stimuli():
    # In Raij 2010, for a single auditory stimuli
    #  Hechsl Gyrus auditory response latency is 23 ms and calcarine cortex visual response latency is 53 ms.

    model = Paredes2022(time_range=(0, 125), theta=16, s=2)
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=1.2,
        auditory_stim_n=1,
        visual_stim_n=0,
        auditory_duration=15,
        visual_duration=20,
        auditory_onset=25,
        visual_onset=25,
    )
    visual_latency = (np.argmax(res.e_.visual_total_input[:, 45]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 45]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 53, atol=18)
    np.testing.assert_allclose(auditory_latency, 23, atol=1)


@pytest.mark.slow
@pytest.mark.model
def test_paredes2022_temporal_filter_visual_stimuli():
    # In Raij 2010, for a single visual stimuli
    # Calcarine cortex visual response latency is 43 ms and Hechsl Gyrus auditory response latency is 82 ms.

    model = Paredes2022(time_range=(0, 125), theta=16, s=2)
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=1.2,
        auditory_stim_n=0,
        visual_stim_n=1,
        auditory_duration=15,
        visual_duration=20,
        auditory_onset=25,
        visual_onset=25,
    )

    visual_latency = (np.argmax(res.e_.visual_total_input[:, 45]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 45]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 43, atol=7)
    np.testing.assert_allclose(auditory_latency, 82, atol=3)


@pytest.mark.slow
@pytest.mark.model
def test_paredes2022_temporal_filter_audiovisual_stimuli():
    # In Raij 2010, for an audiovisual stimuli
    # Calcarine cortex visual response latency is 47 ms and Hechsl Gyrus auditory response latency is 23 ms.

    model = Paredes2022(time_range=(0, 125), theta=16, s=2)
    res = model.run(
        auditory_intensity=1.5,
        visual_intensity=0.85,
        auditory_stim_n=1,
        visual_stim_n=1,
        auditory_duration=15,
        visual_duration=20,
        auditory_onset=25,
        visual_onset=25,
    )
    visual_latency = (np.argmax(res.e_.visual_total_input[:, 45]) - 2500) / 100
    auditory_latency = (
        np.argmax(res.e_.auditory_total_input[:, 45]) - 2500
    ) / 100

    np.testing.assert_allclose(visual_latency, 47, atol=12)
    np.testing.assert_allclose(auditory_latency, 23, atol=1)


@pytest.mark.slow
@pytest.mark.model
def test_paredes2022_fission_illusion():
    model = Paredes2022(time_range=(0, 400))
    res = model.run(
        auditory_intensity=3,
        visual_intensity=1.4,
        auditory_stim_n=2,
        visual_stim_n=1,
        auditory_duration=7,
        visual_duration=12,
        auditory_onset=25,
        visual_onset=25,
        auditory_soa=56,
    )
    fp = findpeaks(method="peakdetect", verbose=0, lookahead=10, interpolate=5)
    X = res.get_modes(include="visual").query("positions==45").visual.values
    results = fp.fit(X)
    visual_peaks_df = results["df"].query("peak==True & valley==False")
    visual_peaks = visual_peaks_df[visual_peaks_df["y"] > 0.15]
    visual_peaks_n = visual_peaks.y.size
    visual_peaks_n

    np.testing.assert_equal(visual_peaks_n, 2)


@pytest.mark.slow
@pytest.mark.model
def test_paredes2022_supraddivity_effect():
    # Two weak stimuli combined are able to elicit a multisensory response.

    model = Paredes2022(time_range=(0, 250))
    res = model.run(
        auditory_intensity=2.4,
        visual_intensity=1.4,
        auditory_stim_n=1,
        visual_stim_n=1,
        auditory_duration=7,
        visual_duration=12,
        auditory_onset=0,
        visual_onset=0,
    )
    fp = findpeaks(method="peakdetect", verbose=0, lookahead=10, interpolate=5)
    X = res.get_modes(include="multi").query("positions==45").multi.values
    results = fp.fit(X)
    multisensory_peaks_df = results["df"].query("peak==True & valley==False")
    multisensory_peaks = multisensory_peaks_df[
        multisensory_peaks_df["y"] > 0.15
    ]
    multisensory_peaks_n = multisensory_peaks.y.size
    multisensory_peaks_n

    np.testing.assert_equal(multisensory_peaks_n, 1)


## TODO Test noise implementation
## TODO Test causes readout / unity report
