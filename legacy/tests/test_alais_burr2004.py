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

from skneuromsi.alais_burr2004 import AlaisBurr2004

# =============================================================================
# Alais and Burr 2004
# =============================================================================


@pytest.mark.parametrize(
    "visual, auditory, expected", [(0, 0, 0), (-5, 5, 0), (-10, 10, 0)]
)
def test_alaisburr2004_run_zero(visual, auditory, expected):
    model = AlaisBurr2004()
    out = model.run(visual_location=visual, auditory_location=auditory)
    idx = out["multisensory"].argmax()
    m_loc = model.possible_locations[idx]

    np.testing.assert_almost_equal(m_loc, expected)


@pytest.mark.parametrize(
    "auditory_sigma, visual_sigma, visual_weight, auditory_weight",
    [(4, 4, 0.5, 0.5), (2, 1, 0.8, 0.2), (8, 4, 0.8, 0.2), (1, 1, 0.5, 0.5)],
)
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

    model = AlaisBurr2004(
        auditory_sigma=auditory_sigma, visual_sigma=visual_sigma
    )

    assert model.visual_weight == visual_weight
    assert model.auditory_weight == auditory_weight
