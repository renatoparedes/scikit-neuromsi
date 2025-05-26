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

"""Tools for scikit-neuromsi neural models."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# ARCHITECTURE TOOLS
# =============================================================================


def calculate_neural_distance(neurons, position_j, position_k):
    """
    Computes the distance between the pre-synaptic and post-synaptic neurons.

    Designed for neurons encoding an external 1D space. It assumes that
    neurons are connected in a circular structure so that every neuron
    receives the same number of lateral connections.

    Parameters
    ----------
    neurons : int
        The number of neurons.
    position_j: float
        Position of the post-synaptic neuron
    position_k: float
        Position of the pre-synaptic neuron

    Returns
    -------
    numpy.float64
        Distance between the two neurons in model units.

    """
    # Evaulates if the distance is higher than the midpoint
    distance = np.abs(position_j - position_k)
    if distance <= neurons / 2:
        return distance
    return neurons - distance


def calculate_lateral_synapses(
    neurons,
    excitation_loc,
    inhibition_loc,
    excitation_scale,
    inhibition_scale,
    dtype=np.float32,
):
    """
    Computes the values of lateral synapses.

    Designed for a group of recurrently connected neurons following a
    “Mexican hat" distribution (a central excitatory zone surrounded by
    an inhibitory annulus) calculated as a substraction of two Gaussians.

    Parameters
    ----------
    neurons : int
        The number of neurons.
    excitation_loc: float
        Loc of the excitatory Gaussian function.
    inhibition_loc: float
        Loc of the inhibitory Gaussian function.
    excitation_scale: float
        Scale of the excitatory Gaussian function.
    inhibition_scale: float
        Scale of the excitatory Gaussian function.
    dtype: numpy class
        Type of the array to store the values.

    Returns
    -------
    numpy.array
        The values of lateral synapses.

    """
    # Creates array to hold the values of the synapses
    the_lateral_synapses = np.zeros((neurons, neurons), dtype)

    # For each pair of neurons, calculate the value of the synapse
    # as a subtraction of Gaussian functions
    for neuron_i in range(neurons):
        for neuron_j in range(neurons):
            if neuron_i == neuron_j:
                the_lateral_synapses[neuron_i, neuron_j] = 0
                continue

            distance = calculate_neural_distance(neurons, neuron_i, neuron_j)
            e_gauss = excitation_loc * np.exp(
                -(np.square(distance)) / (2 * np.square(excitation_scale))
            )
            i_gauss = inhibition_loc * np.exp(
                -(np.square(distance)) / (2 * np.square(inhibition_scale))
            )

            the_lateral_synapses[neuron_i, neuron_j] = e_gauss - i_gauss
    return the_lateral_synapses


def calculate_inter_areal_synapses(neurons, weight, sigma, dtype=np.float32):
    """
    Computes the values of inter-areal synapses.

    Designed for two connected group of neurons following a Gaussian function.
    It assumes symmetrical connectivity and the same number of neurons
    in both groups.

    Parameters
    ----------
    neurons : int
        The number of neurons in each group.
    weight: float
        The highest level of synaptic efficacy.
    sigma: float
        The width of the Gaussian function.
    dtype: numpy class
        Type of the array to store the values.

    Returns
    -------
    numpy.array
        The value of inter-areal synapses.

    """
    # Creates array to hold the values of the synapses
    the_synapses = np.zeros((neurons, neurons), dtype=dtype)

    # For each pair of neurons, calculate the value of the synapse
    # following a Gaussian function
    for j in range(neurons):
        for k in range(neurons):
            d = calculate_neural_distance(neurons, j, k)
            the_synapses[j, k] = weight * np.exp(
                -(np.square(d)) / (2 * np.square(sigma))
            )
    return the_synapses


def prune_synapses(synapses_weight_matrix, pruning_threshold):
    """
    Prunes neural connections.

    Pruning is implemented by assigning zero to those synapses values
    below a given threshold.

    Parameters
    ----------
    synapses_weight_matrix : numpy.array
        Array containing the values of the synapses.
    pruning_threshold: float
        Threshold value of the pruning procedure.

    Returns
    -------
    numpy.array
        The value of the external stimuli for each neuron.

    """
    # Generates a copy of the array
    pruned_synpases = np.copy(synapses_weight_matrix)

    # Prunes synapses below threshold
    pruned_synpases[pruned_synpases < pruning_threshold] = 0

    return pruned_synpases


# =============================================================================
# STIMULI TOOLS
# =============================================================================


def calculate_stimuli_input(
    neurons, intensity, *, scale, loc, dtype=np.float32
):
    """
    Computes the values of stimuli as a spatial Gaussian function.

    The Gaussian accounts for the uncertainty in the detection of stimuli.

    Parameters
    ----------
    neurons : int
        The number of neurons to be stimulated.
    intensity: float
        The strength of the external stimulus.
    scale: float
        Scale of the Gaussian function.
    loc: float
        Loc of the Gaussian function.
    dtype: numpy class
        Type of the array to store the values.

    Returns
    -------
    numpy.array
        The value of the external stimuli for each neuron.

    """
    # Creates array to hold the values of the stimuli
    the_stimuli = np.zeros(neurons, dtype=dtype)

    # Calculates the values of the stimuli as a spatial Gaussian function
    for neuron_j in range(neurons):
        distance = calculate_neural_distance(neurons, neuron_j, loc)
        the_stimuli[neuron_j] = intensity * np.exp(
            -(np.square(distance)) / (2 * np.square(scale))
        )

    return the_stimuli


def create_unimodal_stimuli_matrix(
    neurons,
    stimuli,
    stimuli_duration,
    onset,
    simulation_length,
    time_res,
    dt,
    stimuli_n=1,
    soa=None,
    dtype=np.float32,
):
    """
    Creates the matrix of a unimodal stimuli for each neuron at each timepoint.

    Supports multiple stimuli.

    Parameters
    ----------
    neurons : int
        The number of neurons to be stimulated.
    stimuli: numpy.array
        Array with the values of the stimuli for each neuron.
    stimuli_duration: float
        Duration of the stimuli in model time units.
    onset: float
        Onset of the unimodal stimuli in model time units.
    simulation_length: float
        Total duration of the model run in model time units.
    time_res: float
        Temporal resolution of the model.
    dt: float
        Model integrator dt.
    stimuli_n: int
        Number of unimodal stimuli.
    soa: int
        Stimuli onset asynchrony. Relevant for more than 1 stimuli.
        Must be higher than stimulus_duration.
    dtype: numpy class
        Type of the array to store the values.

    Returns
    -------
    numpy.array
        The value of the external stimuli for each neuron.

    Raises
    ------
    ValueError
        If the total duration of all stimuli plus the total duration of
        the inter-stimulus intervals
        (i.e., `stimuli_duration * stimuli_n + soa * (stimuli_n - 1)`)
        exceeds the `simulation_length`.

    """
    if onset is not None:
        onset = int(onset)

    if stimuli_n is not None:
        stimuli_n = int(stimuli_n)

    if stimuli_duration * stimuli_n > simulation_length:
        raise ValueError("Stimuli total duration exceeds simulation length.")

    if soa is not None:
        soa = int(soa)
        if (
            stimuli_duration * stimuli_n + soa * (stimuli_n - 1)
            > simulation_length
        ):
            raise ValueError(
                "Stimuli total duration exceeds simulation length."
            )
        if soa < stimuli_duration:
            raise ValueError("SOA must be longer than stimulus duration.")

    no_stim = np.zeros(neurons, dtype=dtype)

    if stimuli_n == 0:
        stim = np.tile(no_stim, (simulation_length, 1))
        stimuli_matrix = np.repeat(stim, 1 / time_res, axis=0)
        return stimuli_matrix

    else:
        # Input before onset
        pre_stim = np.tile(no_stim, (onset, 1))

        # Input during stimulus delivery
        stim = np.tile(stimuli, (stimuli_duration, 1))

        # Input during onset asynchrony
        soa_stim = (
            np.tile(no_stim, (soa - stimuli_duration, 1))
            if soa is not None
            else None
        )

        # Input after stimulation
        post_stim_time = (
            simulation_length - onset - stimuli_duration * stimuli_n
        )
        post_stim_time = (
            post_stim_time - (soa - stimuli_duration) * (stimuli_n - 1)
            if soa is not None
            else post_stim_time
        )

        post_stim = np.tile(no_stim, (post_stim_time, 1))

        # Input concatenation
        stim_list = [stim, soa_stim] * (stimuli_n - 1)
        complete_stim = np.vstack(
            (
                pre_stim,
                *stim_list,
                stim,
                post_stim,
            )
        )
        stimuli_matrix = np.repeat(complete_stim, 1 / dt, axis=0)

    return stimuli_matrix


def compute_latency(time, latency):
    """
    Computes the latency-adjusted time in the simulation.

    Latency is computed by subtracting the given latency from the
    current time, ensuring that the result is not negative.

    Parameters
    ----------
    time : float
        The current time in the simulation.
    latency : float
        The latency to subtract from the current time.

    Returns
    -------
    float
        The latency-adjusted time. If the result of the subtraction
        is negative, returns 0 instead.

    Notes
    -----
    This method is used to adjust the time for processing delays
    in the simulation. It ensures that the computed latency does not
    result in a negative time value, which could be invalid for
    certain operations.
    """
    if time - latency >= 0:
        return time - latency
    return 0
