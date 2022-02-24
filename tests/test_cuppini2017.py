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

from skneuromsi.cuppini2017 import Cuppini2017, unisensory_barycenter_readout

# =============================================================================
# CUPPINI 2017
# =============================================================================


@pytest.mark.parametrize("visual, auditory, multi", [(90, 90, 90)])
def test_cuppini2017_run_zero(visual, auditory, multi):
    model = Cuppini2017()
    _, _, m = model.run(
        100, auditory_position=auditory, visual_position=visual
    )
    m_loc = m.argmax()

    np.testing.assert_almost_equal(m_loc, multi)


@pytest.mark.parametrize(
    "auditory, visual, auditory_read, visual_read", [(90, 90, 90, 90)]
)
def test_cuppini2017_unisensory_position_readout(
    auditory, visual, auditory_read, visual_read
):
    model = Cuppini2017()
    a, v, _ = model.run(
        100, auditory_position=auditory, visual_position=visual
    )

    a_loc, v_loc = unisensory_barycenter_readout(180, auditory, visual, a, v)

    np.testing.assert_almost_equal(a_loc, auditory_read, 1)
    np.testing.assert_almost_equal(v_loc, visual_read, 1)


# TODO Get numbers from paper/matlab


@pytest.mark.parametrize(
    "visual_intensity, auditory_intensity, causes",
    [(0, 25, 0), (0, 34, 1), (25, 25, 1)],
)
def test_cuppini2017_unisensory_multisensory_integration(
    visual_intensity, auditory_intensity, causes
):
    model = Cuppini2017()
    _, _, m = model.run(
        100,
        auditory_position=90,
        visual_position=90,
        auditory_intensity=auditory_intensity,
        visual_intensity=visual_intensity,
    )

    n_cause = 0
    if m.max() > 0.2:
        n_cause = 1

    np.testing.assert_equal(n_cause, causes)


# TODO Include tests including noisy stimuli
