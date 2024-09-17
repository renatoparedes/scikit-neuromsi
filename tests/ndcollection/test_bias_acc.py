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

"""test for skneuromsi.ndcollection.bias_acc

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
from skneuromsi.ndcollection import bias_acc, collection


# ==============================================================================
# TESTS
# =============================================================================


@pytest.fixture(scope="session")
def random_ndcollection(random_ndresult, silenced_tqdm_cls):
    def maker(
        *,
        size=2,
        input_modes=2,
        position_coordinates=3,
        run_parameters=None,
        sweep_parameter=None,
        causes=None,
        seed=None,
        **kwargs,
    ):
        random = np.random.default_rng(seed=seed)
        causes = [causes] if np.isscalar(causes) else causes
        runparameters = {} if run_parameters is None else run_parameters

        kwargs.update(
            input_modes_min=input_modes,
            input_modes_max=input_modes,
            position_coordinates_min=position_coordinates,
            position_coordinates_max=position_coordinates,
            seed=random,
        )

        def generator():
            for sweep in range(size):
                curr_causes = random.choice(causes)
                curr_run_parameters = copy.deepcopy(run_parameters)
                if sweep_parameter is not None:
                    curr_run_parameters[sweep_parameter] = sweep
                coll = random_ndresult(
                    causes=curr_causes,
                    run_parameters=curr_run_parameters,
                    **kwargs,
                )
                yield coll

        return collection.NDResultCollection.from_ndresults(
            "collection", generator(), tqdm_cls=silenced_tqdm_cls
        )

    return maker


def test_BiasAcc_bias(random_ndcollection):

    random_ndcollection()
    bias = bias_acc.NDResultCollectionBiasAcc(coll, silenced_tqdm_cls)

    import ipdb

    ipdb.set_trace()
