#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# This code was ripped of from scikit-criteria on 05-aug-2024.
# https://github.com/quatrope/scikit-criteria/blob/5f829f7e18129b76f4ddfc89e1
# a6bea2f1a31306/tests/utils/test_dict_cmp.py
# Util this point the copyright is
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skneuromsi.utils.dict_cmp

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


from skneuromsi.utils import dict_cmp


# =============================================================================
# The tests
# =============================================================================


def test_dict_allclose():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": {}}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": {}}
    assert dict_cmp.dict_allclose(left, right)


def test_dict_allclose_same_obj():
    dict0 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    assert dict_cmp.dict_allclose(dict0, dict0)


def test_dict_allclose_different_keys():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "d": 1}
    assert dict_cmp.dict_allclose(left, right) is False


def test_dict_allclose_different_types():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1.0}
    assert dict_cmp.dict_allclose(left, right) is False
