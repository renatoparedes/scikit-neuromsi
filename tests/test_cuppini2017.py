# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skneuromsi.cuppini2017 import Cuppini2017

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
