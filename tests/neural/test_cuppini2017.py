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

import pytest

from skneuromsi.neural import Cuppini2017

# =============================================================================
# CUPPINI 2017
# =============================================================================


@pytest.mark.parametrize("visual, auditory, multi", [(90, 90, 90)])
@pytest.mark.slow
@pytest.mark.model
def test_cuppini2017_run_zero(visual, auditory, multi):
    model = Cuppini2017()
    result = model.run(
        auditory_position=auditory,
        visual_position=visual,
    )

    max_time = result.to_xarray()["times"].max().values
    m_loc = (
        result.get_modes(include="multi")
        .query(f"times=={max_time}")
        .values.argmax()
    )
    np.testing.assert_almost_equal(m_loc, multi)


@pytest.mark.parametrize(
    "visual_intensity, auditory_intensity, causes",
    [(0, 25, 0), (0, 34, 1), (25, 25, 1)],
)
@pytest.mark.slow
@pytest.mark.model
def test_cuppini2017_unisensory_multisensory_integration(
    visual_intensity, auditory_intensity, causes
):
    model = Cuppini2017()
    result = model.run(
        auditory_position=90,
        visual_position=90,
        auditory_intensity=auditory_intensity,
        visual_intensity=visual_intensity,
    )

    n_cause = 0
    if result.get_modes(include="multi").multi.max() > 0.2:
        n_cause = 1

    np.testing.assert_equal(n_cause, causes)


# TODO Include tests including noisy stimuli
