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

from skneuromsi.bayesian import Zhu2024

# =============================================================================
# Zhu 2024
# =============================================================================


@pytest.mark.parametrize(
    "visual_numerosity_sigma, prob_1_flash_expected, prob_2_flash_expected",
    [
        (0.3, 0.875, 0.075),
        (0.6, 0.575, 0.325),
        (0.9, 0.45, 0.375),
        (1.2, 0.35, 0.425),
    ],
)
@pytest.mark.slow
@pytest.mark.model
def test_zhu2024_run(
    visual_numerosity_sigma, prob_1_flash_expected, prob_2_flash_expected
):
    """
    Reproduces the effect observed in Figure 1A from Zhu et al. (2024).
    The test simulates the fission illusion condition (F1B2, one flash paired
    with two beeps) with Pcommon = 0.6, σA = 0.6, σp = 3, μp = 1.5,
    and σV varied from 1.2 to 0.3 (step = 0.3). The results show that
    susceptibility to illusion decreases with increasing visual precision.
    Parameters not reported in the original paper were approximated.
    """

    bci_model = Zhu2024(
        n=1000000,
        time_range=(0, 1000),
        time_res=1,
        numerosity_range=(0, 4),
        numerosity_res=1,
    )

    out = bci_model.run(
        visual_time=500,
        auditory_time=675,
        auditory_numerosity=2,
        visual_numerosity=1,
        auditory_numerosity_sigma=0.60,
        visual_numerosity_sigma=visual_numerosity_sigma,
        noise=True,
        prior_time_sigma=1000.0,
        prior_time_mu=500.0,
        prior_numerosity_mu=1.5,
        prior_numerosity_sigma=3.0,
        p_common=0.6,
    )

    visual_numerosity_estimate = out.e_["visual_numerosity"]
    prob_1_flash = visual_numerosity_estimate[1]
    prob_2_flash = visual_numerosity_estimate[2]

    np.testing.assert_approx_equal(
        prob_1_flash, prob_1_flash_expected, significant=1
    )
    np.testing.assert_approx_equal(
        prob_2_flash, prob_2_flash_expected, significant=1
    )
