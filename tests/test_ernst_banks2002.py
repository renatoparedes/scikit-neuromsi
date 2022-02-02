#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-neuromsi Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skneuromsi.ernst_banks2002 import ErnstBanks2002

# =============================================================================
# Ernst and Banks 2002
# =============================================================================


@pytest.mark.parametrize(
    "visual, haptic, expected", [(55, 55, 55), (53, 57, 55), (51, 59, 55)]
)
def test_ernstbanks2002_run_average_height(visual, haptic, expected):
    model = ErnstBanks2002()
    out = model.run(visual_height=visual, haptic_height=haptic)
    idx = out["multisensory"].argmax()
    m_loc = model.possible_heights[idx]

    np.testing.assert_almost_equal(m_loc, expected)


@pytest.mark.parametrize(
    "haptic_sigma, visual_sigma, visual_weight, haptic_weight",
    [(4, 4, 0.5, 0.5), (2, 1, 0.8, 0.2), (8, 4, 0.8, 0.2), (1, 1, 0.5, 0.5)],
)
def test_ernstbanks2002_mle_integration(
    haptic_sigma, visual_sigma, visual_weight, haptic_weight
):
    """
    Data From:
        M. O. Ernst and M. S. Banks, “Humans integrate
        visual and haptic information in a statistically
        optimal fashion,” Nature, vol. 415, no. 6870,
        pp. 429-433, Jan. 2002, doi: 10.1038/415429a.
    """

    model = ErnstBanks2002(
        haptic_sigma=haptic_sigma, visual_sigma=visual_sigma
    )

    assert model.visual_weight == visual_weight
    assert model.haptic_weight == haptic_weight
