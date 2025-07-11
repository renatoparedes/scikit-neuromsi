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

"""Implementation of bias analysis in multisensory integration.

The NDResultBiasAcc class provides an accessor, NDResultBiasAcc, for analyzing
biases in the context of multisensory integration. It offers methods to
calculate biases and mean biases based on specified influence and changing
parameters.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import methodtools

import numpy as np

import pandas as pd

from ..utils import AccessorABC

# =============================================================================
# BIAS ACC
# =============================================================================


class NDResultCollectionBiasAcc(AccessorABC):
    """Accessor for calculating biases in an NDResultCollection.

    Bias analysis in multisensory integration refers to the examination and
    identification of potential biases that may influence the way different
    sensory modalities are combined or integrated in the human brain.

    This accessor provides methods for calculating biases based on
    various parameters.

    Parameters
    ----------
    ndcollection : NDResultCollection
        The NDResultCollection for which to calculate biases.
    tqdm_cls : tqdm.tqdm, optional
        The tqdm class to use. Defaults to None.


    """

    _default_kind = "bias"

    def __init__(self, ndcollection, tqdm_cls):
        self._nd_collection = ndcollection
        self._tqdm_cls = tqdm_cls

    def _bias_as_frame(
        self, disp_mtx, influence_parameter, changing_parameter, bias_arr
    ):
        """Create a DataFrame from the calculated biases.

        Parameters
        ----------
        disp_mtx : pandas.DataFrame
            Disparity matrix containing the changing and influence parameters.
        influence_parameter : str
            Parameter being influenced by the cross-modal bias.
        changing_parameter : str
            Parameter changing across iterations.
        bias_arr : numpy.ndarray
            Array of calculated biases.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representing biases with columns representing changing
            and influence parameters.

        """
        # The new index is obtained by subtracting the relative value of the
        # changing parameter from the influence parameter.
        dispm_diff = (
            disp_mtx[changing_parameter] - disp_mtx[influence_parameter]
        )

        # Now we need to know how many repetitions there are in our biases
        cpd_len = len(disp_mtx)
        cpd_unique_len = len(np.unique(disp_mtx[changing_parameter]))

        repeat = cpd_len // cpd_unique_len
        pos = cpd_len // repeat

        # Create the DataFrame
        bias_df = pd.DataFrame(bias_arr.reshape(pos, repeat))

        # The index is repeated, so we only want one each 'repeat'.
        bias_df.index = pd.Index(dispm_diff.values[::repeat], name="Disparity")

        # Create a multi-level column
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
        """Calculate biases for a specified influence parameter.

        This method calculates biases in the context of multisensory
        integration analysis. Biases represent the deviation of a specific
        parameter's influence on the integration process across different
        iterations. The analysis considers the relationship between the
        changing parameter and the influence parameter in each iteration over
        a given mode.

        Parameters
        ----------
        influence_parameter : str
            The parameter being influenced by the cross-modal bias.
        changing_parameter : str or None, optional
            The parameter changing across iterations. If None,
            automatically selected.
        dim : str or None, optional
            The dimension for calculating biases. If None, defaults to 'time'.
        mode : str or None, optional
            The mode for which to calculate biases. If None, defaults to the
            mode with maximum variance.
        quiet : bool, optional
            If True, suppress tqdm progress bar. Defaults to False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing biases with columns representing changing
            and influence parameters.
            The columns represent the changing and influence parameters, and
            each row corresponds to a specific disparity in the relationship
            between these parameters across iterations.

        Raises
        ------
        ValueError
            If the specified parameters are invalid.

        """
        ndresults = self._nd_collection
        tqdm_cls = self._tqdm_cls

        mode = ndresults.coerce_mode(mode)
        changing_parameter = ndresults.coerce_parameter(changing_parameter)
        dim = ndresults.coerce_dimension(dim)

        # unchanged_parameters = ~nd_collection.changing_parameters()
        # if not unchanged_parameters[influence_parameter]:
        #    raise ValueError(
        #        f"influence_parameter {influence_parameter!r} are not fixed"
        #    )

        disp_mtx = ndresults.disparity_matrix()[
            [changing_parameter, influence_parameter]
        ]

        influence_value = disp_mtx[influence_parameter][0]

        # tqdm progress bar
        if quiet is False and tqdm_cls is not None:
            ndresults = tqdm_cls(iterable=ndresults, desc="Calculating biases")

        bias_arr = np.zeros(len(ndresults))
        for idx, res in enumerate(ndresults):
            ref_value = res.run_parameters[changing_parameter]

            # here we extract all the values of the mode we are interested in
            modes_values = res.get_modes(include=mode)

            # we determine the values of the mode in the dimension
            # that interests us
            max_dim_index = modes_values.index.get_level_values(dim).max()
            max_dim_values = modes_values.xs(
                max_dim_index, level=dim
            ).values.T[0]
            current_dim_position = max_dim_values.argmax()

            bias = np.abs(current_dim_position - influence_value) / np.abs(
                influence_value - ref_value
            )

            bias_arr[idx] = bias

        # convert biases array to a DataFrame
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
        """Calculate the mean biases in multisensory integration analysis.

        This method calculates the mean biases in the context of multisensory
        integration analysis. Mean biases represent the average deviation of a
        specific parameter's influence on the integration process across
        different iterations. The analysis considers the relationship between
        the changing parameter and the influence parameter in each iteration.

        Parameters
        ----------
        influence_parameter : str
            The parameter being influenced by the cross-modal biases.
        changing_parameter : str or None, optional
            The parameter that changes across iterations. If None, the function
            automatically selects it.
        dim : str or None, optional
            The dimension for calculating mean biases. If None, it defaults to
            'time'.
        mode : str or None, optional
            The mode for which to calculate mean biases. If None, it defaults
            to the mode with maximum variance.
        quiet : bool, optional
            If True, suppresses the tqdm progress bar. Defaults to False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the mean biases in multisensory integration
            analysis. The columns represent the changing and influence
            parameters, and each row corresponds to a specific disparity in
            the relationship between these parameters across iterations.

        Raises
        ------
        ValueError
            If the specified parameters are invalid or the influence parameter
            is not fixed across iterations.

        """
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
