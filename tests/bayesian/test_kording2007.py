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

from skneuromsi.bayesian import Kording2007

# =============================================================================
# Kording 2007
# =============================================================================


@pytest.mark.parametrize(
    "auditory, visual, auditory_expected, visual_expected",
    [(-10, 10, -9.43, -4.29)],
)
@pytest.mark.slow
@pytest.mark.model
def test_kording2007_run(visual, auditory, visual_expected, auditory_expected):
    """
    Data From:
        M. Samad, K. Sita, A. Wang and L. Shams. Bayesian Causal Inference
        Toolbox (BCIT) for MATLAB.
        https://shamslab.psych.ucla.edu/bci-matlab-toolbox/
    """

    position_range = (-42, 42)
    locations = np.linspace(
        position_range[0], position_range[1], 50, retstep=True
    )

    model = Kording2007(n=1000000, position_range=position_range)
    out = model.run(
        visual_position=visual,
        auditory_position=auditory,
    )
    a_idx = out.get_modes("auditory").auditory.argmax()
    v_idx = out.get_modes("visual").visual.argmax()

    a_loc = locations[0][a_idx]
    v_loc = locations[0][v_idx]

    np.testing.assert_almost_equal(a_loc, auditory_expected, 2)
    np.testing.assert_almost_equal(v_loc, visual_expected, 2)
