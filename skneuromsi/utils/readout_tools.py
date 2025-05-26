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

"""Tools to read the output of scikit-neuromsi models."""

# =============================================================================
# IMPORTS
# =============================================================================

import itertools

import numpy as np

from scipy.signal import find_peaks

# =============================================================================
# TOOLS
# =============================================================================


def calculate_multiple_peaks_probability(peaks_values):
    """
    Computes the probability of reading out multiple stimuli from the ouput.

    The probability of perceiving two stimuli is defined as the product of
    the two peak values. When more than two peak values are provided, the
    average product of all possible combinations of peak values is computed.

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
    Computes the probability of reading a unique stimulus from the model ouput.

    The probability of perceiving one stimulus is defined as the single peak
    value. When two or more peak values are provided, the complementary
    probability of perceiving multiple stimuli is computed.

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
    peaks_values = np.asarray(peaks_values)

    # If no peaks were found, assign 0
    if peaks_values.size == 0:
        single_peak_probability = 0.0

    # If one peak was found, assign the peak value
    elif peaks_values.size == 1:
        single_peak_probability = peaks_values.item()

    # If multiple peaks were found, assign the complementary
    # probability of multiple peaks
    else:
        single_peak_probability = 1.0 - calculate_multiple_peaks_probability(
            peaks_values
        )

    return single_peak_probability


def calculate_causes_from_peaks(
    mode_activity_data,
    causes_kind="count",
    peak_threshold=0.15,
    peak_distance=None,
):
    """
    Computes the number of causes from peaks found in modal activity.

    The peaks are identified using the peakdetect method. The algorithm
    requires to set the lookahead parameter, which is the distance to look
    ahead from a peak candidate to determine if it is the actual peak.

    Parameters
    ----------
    mode_activity_data : numpy.array
        Array containing the activity of a mode in space or time.
    causes_kind : str
        The name of the causes readout method.

        - 'count' : counts the number of peaks (default)
        - 'prob' : takes the height of the peaks as probability values.
    peak_threshold : float
        The minimum peak height to detect peaks.

    Returns
    -------
    numpy.float64
        Causes identified from modal activity.

    """
    # Find the peaks in the data
    peaks, peaks_props = find_peaks(
        mode_activity_data,
        height=peak_threshold,
        prominence=peak_threshold,
        distance=peak_distance,
    )

    # Determine the type of cause to calculate
    if causes_kind == "count":
        # If counting the number of causes, assign the number of peaks found
        causes = len(peaks)
    elif causes_kind == "prob":
        # If calculating the probability of a unique cause,
        # calculate the probability of detecting a single peak
        peaks_values = peaks_props["peak_heights"]
        causes = calculate_single_peak_probability(peaks_values)
    else:
        # If no valid cause type is specified, assign None
        causes = None

    return causes


def calculate_spatiotemporal_causes_from_peaks(
    mode_spatiotemporal_activity_data,
    causes_kind="count",
    causes_dim="space",
    time_point=-1,
    spatial_point=0,
    peak_threshold=0.15,
    peak_distance=None,
):
    """
    Computes the number of causes from peaks found in modal activity.

    The peaks are identified using the peakdetect method. The algorithm
    requires to set the lookahead parameter, which is the distance to look
    ahead from a peak candidate to determine if it is the actual peak.

    Parameters
    ----------
    mode_spatiotemporal_activity_data : 2D numpy.array
        Array containing the activity of a mode in space and time.
        Assumes that temporal dimension is organised in rows and
        spatial dimension in columns.
    causes_kind : str
        The name of the causes readout method to employ.
    causes_dim : str
        The name of the dimension to readout from.

        - 'space' : reads causes from spatial dimension (default).
        - 'time' : reads causes from temporal dimension.
    time_point: int
        The temporal point where spatial readout is computed.
    position_point: int
        The spatial point where temporal readout is computed.
    peak_threshold : float
        The minimum persistence score to detect peaks.
    lookahead: int
        The distance to look ahead from a peak candidate.

    Returns
    -------
    numpy.float64
        Causes identified from modal activity in the desired dimension.

    """
    # Determine the dimension of interest
    if causes_dim == "space":
        # If calculating spatial causes, get the data from
        # time point the multi matrix
        mode_activity_data = mode_spatiotemporal_activity_data[time_point, :]

    elif causes_dim == "time":
        # If calculating temporal causes, get data from position
        # oint in multi matrix
        mode_activity_data = mode_spatiotemporal_activity_data[
            :, spatial_point
        ]
    else:
        # If not valid dimension defined, return None
        return None

    # Calculate causes in the selected dimension
    causes = calculate_causes_from_peaks(
        mode_activity_data=mode_activity_data,
        causes_kind=causes_kind,
        peak_threshold=peak_threshold,
        peak_distance=peak_distance,
    )

    # Return the calculated causes
    return causes
