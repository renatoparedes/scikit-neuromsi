import numpy as np

import pytest

from skneuromsi.alais_burr2004 import AlaisBurr2004


@pytest.mark.parametrize(
    "visual, auditory, expected", [(0, 0, 0), (-5, 5, 0), (-10, 10, 0)]
)
def test_AlaisBurr2004_run_zero(visual, auditory, expected):
    model = AlaisBurr2004()
    out = model.run(visual_location=visual, auditory_location=auditory)
    idx = out["multisensory_estimator"].argmax()
    m_loc = model.posible_locations[idx]

    np.testing.assert_almost_equal(m_loc, expected)


@pytest.mark.parametrize(
    "auditory_sigma, visual_sigma, visual_weight, auditory_weight",
    [(4, 4, 0.5, 0.5), (2, 1, 0.8, 0.2), (8, 4, 0.8, 0.2), (1, 1, 0.5, 0.5)],
)
def test_AlaisBurr2004_MLE_integration(
    auditory_sigma, visual_sigma, visual_weight, auditory_weight
):
    model = AlaisBurr2004(
        auditory_sigma=auditory_sigma, visual_sigma=visual_sigma
    )

    assert model.visual_weight == visual_weight
    assert model.auditory_weight == auditory_weight
