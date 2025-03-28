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

"""test for skneuromsi.ndcollection.bias_acc

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from skneuromsi.ndcollection import bias_acc


# =============================================================================
# TESTS
# =============================================================================


def test_NDResultCollectionBiasAcc_bias(random_ndcollection):
    coll = random_ndcollection(
        size=10, seed=42, run_parameters={"p0": 1}, sweep_parameter="p1"
    )
    bias = bias_acc.NDResultCollectionBiasAcc(coll, tqdm_cls=coll.tqdm_cls)

    expected = pd.DataFrame(
        [
            [54.0],
            [np.inf],
            [50.0],
            [23.5],
            [4.0],
            [12.75],
            [2.0],
            [6.5],
            [8.285714285714286],
            [4.625],
        ],
        index=pd.Index([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], name="Disparity"),
        columns=pd.MultiIndex.from_tuples(
            [("p1", "p0", 0)],
            names=["Changing parameter", "Influence parameter", "Iteration"],
        ),
    )
    pd.testing.assert_frame_equal(bias.bias("p0"), expected)


def test_NDResultCollectionBiasAcc_bias_mean(random_ndcollection):
    coll = random_ndcollection(
        size=10, seed=42, run_parameters={"p0": 1}, sweep_parameter="p1"
    )
    bias = bias_acc.NDResultCollectionBiasAcc(coll, tqdm_cls=coll.tqdm_cls)

    expected = pd.DataFrame(
        [
            [54.0],
            [np.inf],
            [50.0],
            [23.5],
            [4.0],
            [12.75],
            [2.0],
            [6.5],
            [8.285714285714286],
            [4.625],
        ],
        index=pd.Index([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], name="Disparity"),
        columns=pd.MultiIndex.from_tuples(
            [("p1", "p0", "mean")],
            names=["Changing parameter", "Influence parameter", "Bias"],
        ),
    )
    pd.testing.assert_frame_equal(bias.bias_mean("p0"), expected)
