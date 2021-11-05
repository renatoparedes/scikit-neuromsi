import numpy as np

import pytest

from skneuromsi.kording2007 import Kording2007


@pytest.mark.parametrize(
    "auditory, visual, auditory_expected, visual_expected",
    [(-10, 10, -9.43, -4.29)],
)
def test_Kording2007_run(visual, auditory, visual_expected, auditory_expected):
    model = Kording2007()
    out = model.run(visual_location=visual, auditory_location=auditory)
    a_idx = out["auditory"].argmax()
    v_idx = out["visual"].argmax()

    a_loc = model.possible_locations[0][a_idx]
    v_loc = model.possible_locations[0][v_idx]

    np.testing.assert_almost_equal(a_loc, auditory_expected, 2)
    np.testing.assert_almost_equal(v_loc, visual_expected, 2)


# Test for prob coupled and bias, as in Kording 2007.
# Test single stimuli and different strategies.
