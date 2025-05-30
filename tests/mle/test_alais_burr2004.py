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

from skneuromsi.mle import AlaisBurr2004

# =============================================================================
# Alais and Burr 2004
# =============================================================================


@pytest.mark.parametrize(
    "visual, auditory, expected", [(0, 0, 0), (-5, 5, 0), (-10, 10, 0)]
)
@pytest.mark.slow
@pytest.mark.model
def test_alaisburr2004_run_zero(visual, auditory, expected):
    position = (-20, 20)
    position_res = 0.01
    locations = np.arange(position[0], position[1], position_res)

    model = AlaisBurr2004(position_range=position, position_res=position_res)
    out = model.run(
        visual_position=visual,
        auditory_position=auditory,
    )

    idx = out.get_modes(include="multi").multi.argmax()
    m_loc = locations[idx]

    np.testing.assert_almost_equal(m_loc, expected)


@pytest.mark.parametrize(
    "auditory_sigma, visual_sigma, visual_weight, auditory_weight",
    [(4, 4, 0.5, 0.5), (2, 1, 0.8, 0.2), (8, 4, 0.8, 0.2), (1, 1, 0.5, 0.5)],
)
@pytest.mark.slow
@pytest.mark.model
def test_alaisburr2004_mle_integration(
    auditory_sigma, visual_sigma, visual_weight, auditory_weight
):
    """
    Data From:
        M. O. Ernst and M. S. Banks, “Humans integrate
        visual and haptic information in a statistically
        optimal fashion,” Nature, vol. 415, no. 6870,
        pp. 429-433, Jan. 2002, doi: 10.1038/415429a.
    """

    model = AlaisBurr2004()

    model_visual_weight = model.weight_calculator(visual_sigma, auditory_sigma)
    model_auditory_weight = model.weight_calculator(
        auditory_sigma, visual_sigma
    )

    assert model_visual_weight == visual_weight
    assert model_auditory_weight == auditory_weight
