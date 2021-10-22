import numpy as np

from skneuromsi.alais_burr2004 import AlaisBurr2004


def test_AlaisBurr2004_run_zero():
    model = AlaisBurr2004()
    out = model.run(visual_location=0, auditory_location=0)

    np.testing.assert_almost_equal(out.mean(), 0.0, 0)
