#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for skneuromsi.utils.decorator"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

import pytest

from skneuromsi.utils import AccessorABC


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_AccessorABC():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self, v):
            self._v = v

        def zaraza(self):
            return self._v

    acc = FooAccessor(np.random.random())
    assert acc("zaraza") == acc.zaraza() == acc()


def test_AccessorABC_no__default_kind():
    with pytest.raises(TypeError):

        class FooAccessor(AccessorABC):
            pass

    with pytest.raises(TypeError):
        AccessorABC()


def test_AccessorABC_invalid_kind():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self):
            self.dont_work = None

        def _zaraza(self):
            pass

    acc = FooAccessor()

    with pytest.raises(ValueError):
        acc("_zaraza")

    with pytest.raises(ValueError):
        acc("dont_work")
