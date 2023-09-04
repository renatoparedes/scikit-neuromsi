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

""""""

# =============================================================================
# IMPORTS
# =============================================================================

import methodtools

import numpy as np

import pandas as pd

from ...utils import AccessorABC


# =============================================================================
# BIAS ACC
# =============================================================================


class NDResultBiasAcc(AccessorABC):
    _default_kind = "bias"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def _bias_as_frame(
        self, disp_mtx, influence_parameter, changing_parameter, bias_arr
    ):

        # el nuevo indice va a ser sale del valor relativo del
        # parametro cambia "menos" el que influye
        dispm_diff = (
            disp_mtx[changing_parameter] - disp_mtx[influence_parameter]
        )

        # ahora necesitamos saber cuantas repetisiones hay en nuestros bias
        cpd_len = len(disp_mtx)
        cpd_unique_len = len(np.unique(disp_mtx[changing_parameter]))

        repeat = cpd_len // cpd_unique_len
        pos = cpd_len // repeat

        # creamos el dataframe
        bias_df = pd.DataFrame(bias_arr.reshape(pos, repeat))

        # el indice esta repetido, por lo que solo queremos uno cada 'repeat'.
        bias_df.index = pd.Index(dispm_diff.values[::repeat], name="Disparity")

        # cramos un columna multi nivel
        cnames = ["Changing parameter", "Influence parameter", "Iteration"]
        cvalues = [
            (changing_parameter, influence_parameter, it)
            for it in bias_df.columns
        ]
        bias_df.columns = pd.MultiIndex.from_tuples(cvalues, names=cnames)

        return bias_df

    @methodtools.lru_cache(maxsize=None)
    def bias(
        self,
        influence_parameter,
        *,
        changing_parameter=None,
        dim=None,
        mode=None,
        quiet=False,
    ):
        nd_collection = self._nd_collection

        mode = nd_collection.coerce_mode(mode)
        changing_parameter = nd_collection.coerce_parameter(changing_parameter)
        dim = nd_collection.coerce_dimension(dim)

        unchanged_parameters = ~nd_collection.changing_parameters()
        if not unchanged_parameters[influence_parameter]:
            raise ValueError(
                f"influence_parameter {influence_parameter!r} are not fixed"
            )

        disp_mtx = nd_collection.disparity_matrix()[
            [changing_parameter, influence_parameter]
        ]

        influence_value = disp_mtx[influence_parameter][0]

        ndresults = nd_collection._ndresults
        progress_cls = nd_collection._progress_cls
        if quiet is False and progress_cls is not None:
            ndresults = progress_cls(
                iterable=ndresults, desc="Calculating biases"
            )

        bias_arr = np.zeros(len(nd_collection))
        for idx, res in enumerate(ndresults):

            ref_value = res.run_params[changing_parameter]

            # aca sacamos todos los valores del modo que nos interesa
            modes_values = res.get_modes(mode)

            # determinamos los valores del modo en la dimension
            # que nos interesa
            max_dim_index = modes_values.index.get_level_values(dim).max()
            max_dim_values = modes_values.xs(
                max_dim_index, level=dim
            ).values.T[0]
            current_dim_position = max_dim_values.argmax()

            bias = np.abs(current_dim_position - ref_value) / np.abs(
                ref_value - influence_value
            )

            bias_arr[idx] = bias

        # convertimos los bias_arr en un dataframe
        bias_df = self._bias_as_frame(
            disp_mtx, influence_parameter, changing_parameter, bias_arr
        )

        return bias_df

    def bias_mean(
        self,
        influence_parameter,
        *,
        changing_parameter=None,
        dim=None,
        mode=None,
        quiet=False,
    ):
        bias = self.bias(
            influence_parameter=influence_parameter,
            changing_parameter=changing_parameter,
            dim=dim,
            mode=mode,
            quiet=quiet,
        )
        mbias_df = bias.mean(axis=1).to_frame()

        cnames = bias.columns.names[:-1] + ["Bias"]
        cvalues = bias.columns[0][:-1] + ("mean",)
        mbias_df.columns = pd.MultiIndex.from_tuples([cvalues], names=cnames)

        return mbias_df
