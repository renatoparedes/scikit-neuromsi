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

"""Tools to read the output of scikit-neuromsi models."""

# =============================================================================
# IMPORTS
# =============================================================================

import itertools
import numpy as np

# =============================================================================
# TOOLS
# =============================================================================


def calculate_multiple_peaks_probability(peaks_values):
    """
    Computes the probability of reading out two or more stimuli
    from the model ouput. The probability of perceiving two stimuli
    is defined as the product of the two peak values.
    When more than two peak values are provided, the average
    product of all possible combinations of peak values is computed.

    Parameters
    ----------
    peak_values : np.array
        Array containing the peak values detected in the model output.
        Values are expected to be scaled from 0 to 1.

    Returns
    -------
    numpy.float64
        Probability of reading out multiple peaks (stimuli).

    """
    # Gets all possible combinations of two or more peak values
    combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations(peaks_values, i + 2)
            for i in range(len(peaks_values))
        )
    )

    # Calculates the product of all combinations
    probs_array = np.array([])
    for i in combinations:
        probs_array = np.append(probs_array, np.array(i).prod())

    # Calculates the average product of all combinations
    multiple_peaks_probability = probs_array.sum() / probs_array.size

    return multiple_peaks_probability


def calculate_single_peak_probability(peaks_values):
    """
    Computes the probability of reading a unique stimulus
    from the model ouput. The probability of perceiving one stimulus
    is defined as the single peak value. When two or more peak values
    are provided, the complementary probability of perceiving
    multiple stimuli is computed.

    Parameters
    ----------
    peak_values : numpy.array
        Array containing the peak values detected in the model output.
        Values are expected to be scaled from 0 to 1.

    Returns
    -------
    numpy.float64
        Probability of reading out a single peak (stimulus).

    """
    # If no peaks were found, assign 0
    if peaks_values.size == 0:
        single_peak_probability = 0.0

    # If one peak was found, assign the peak value
    elif peaks_values.size == 1:
        single_peak_probability = peaks_values.item()

    # If multiple peaks were found, assign the complementary probability of multiple peaks
    else:
        single_peak_probability = 1.0 - calculate_multiple_peaks_probability(
            peaks_values
        )

    return single_peak_probability
