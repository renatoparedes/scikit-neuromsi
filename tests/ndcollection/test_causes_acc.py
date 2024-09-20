#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for skneuromsi.ndcollection.causes_acc

"""

# =============================================================================
# IMPORTS
# =============================================================================

import io
from unittest import mock

import numpy as np

import pandas as pd

import pytest

import skneuromsi as sknmsi
from skneuromsi.ndcollection import bias_acc, causes_acc, collection, cplot_acc

# =============================================================================
# TESTS
# =============================================================================


def test_NDResultCollectionCausesAcc_causes_by_parameter(random_ndcollection):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    expected = pd.DataFrame(
        [
            [0, 0],
            [1, 0],
            [2, 1],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 1],
            [7, 0],
            [8, 1],
            [9, 0],
        ],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Iteration"),
        columns=pd.MultiIndex.from_tuples(
            [("Parameters", "p1"), ("", "Causes")]
        ),
    )

    pd.testing.assert_frame_equal(
        causes.causes_by_parameter(parameter="p1"), expected
    )


def test_NDResultCollectionCausesAcc_unique_causes(random_ndcollection):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    np.testing.assert_array_equal(causes.unique_causes(parameter="p1"), [0, 1])


def test_NDResultCollectionCausesAcc_n_report(random_ndcollection):
    coll = coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    expected = pd.DataFrame(
        [[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Disparity"),
        columns=pd.Index(["p1"], name="Causes"),
    )

    pd.testing.assert_frame_equal(causes.n_report(0, parameter="p1"), expected)

    expected = pd.DataFrame(
        [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Disparity"),
        columns=pd.Index(["p1"], name="Causes"),
    )

    pd.testing.assert_frame_equal(causes.n_report(1, parameter="p1"), expected)


def test_NDResultCollectionCausesAcc_unity_report(random_ndcollection):
    coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    expected = pd.DataFrame(
        [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Disparity"),
        columns=pd.Index(["p1"], name="Causes"),
    )

    pd.testing.assert_frame_equal(
        causes.unity_report(parameter="p1"), expected
    )


def test_NDResultCollectionCausesAcc_mean_report(random_ndcollection):
    coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    expected = pd.DataFrame(
        [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Disparity"),
        columns=pd.Index(["p1"], name="Causes"),
    )

    pd.testing.assert_frame_equal(causes.mean_report(parameter="p1"), expected)


def test_NDResultCollectionCausesAcc_describe_causes(random_ndcollection):
    coll = random_ndcollection(
        size=10,
        seed=42,
        run_parameters={"p0": 1},
        sweep_parameter="p1",
        causes=[0, 1],
    )
    causes = causes_acc.NDResultCollectionCausesAcc(coll)

    expected = pd.DataFrame(
        [
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="Disparity"),
        columns=pd.MultiIndex.from_product(
            [
                ("p1",),
                ("count", "mean", "std", "min", "25%", "50%", "75%", "max"),
            ],
            names=["Causes", None],
        ),
    )

    pd.testing.assert_frame_equal(
        causes.describe_causes(parameter="p1"), expected
    )
