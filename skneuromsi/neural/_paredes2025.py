#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


import copy
from dataclasses import dataclass

from brainpy import odeint

import numpy as np

from ..core import SKNMSIMethodABC
from ..utils.neural_tools import (
    calculate_inter_areal_synapses,
    calculate_lateral_synapses,
    calculate_stimuli_input,
    compute_latency,
    create_unimodal_stimuli_matrix,
    prune_synapses,
)
from ..utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


@dataclass
class Paredes2025Integrator:
    """A class representing the integrator for the Paredes2025 model."""

    tau: tuple
    s: float
    theta: float
    name: str = "Paredes2025Integrator"

    @property
    def __name__(self):
        """Return the name of the Integrator."""
        return self.name

    def sigmoid(self, u):
        """
        Computes the sigmoid activation function.

        Parameters
        ----------
        u : float or np.ndarray
            The input to the sigmoid function.

        Returns
        -------
        float or np.ndarray
            The result of the sigmoid function applied to `u`.
        """
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def __call__(self, y_a, y_v, y_m, t, u_a, u_v, u_m):
        """
        Computes the activities of neurons.

        Parameters
        ----------
        y_a : np.ndarray
            The current state of the auditory layer neurons.
        y_v : np.ndarray
            The current state of the visual layer neurons.
        y_m : np.ndarray
            The current state of the multisensory layer neurons.
        t : float
            The current time in the simulation.
        u_a : np.ndarray
            The total input to the auditory layer neurons.
        u_v : np.ndarray
            The total input to the visual layer neurons.
        u_m : np.ndarray
            The total input to the multisensory layer neurons.

        Returns
        -------
        tuple
            A tuple containing the activities of neurons.
        """
        # Auditory
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau)

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau)

        # Multisensory
        dy_m = (-y_m + self.sigmoid(u_m)) * (1 / self.tau)

        return dy_a, dy_v, dy_m


@dataclass
class Paredes2025TemporalFilter:
    """Temporal filter for the Paredes2025 model."""

    tau: tuple
    name: str = "Paredes2025TemporalFilter"

    @property
    def __name__(self):
        """Return the name of the Temporal Filter."""
        return self.name

    def __call__(
        self,
        a_outside_input,
        v_outside_input,
        m_outside_input,
        auditoryfilter_input,
        visualfilter_input,
        multisensoryfilter_input,
        t,
        a_external_input,
        v_external_input,
        m_external_input,
        a_cross_modal_input,
        v_cross_modal_input,
        a_feedback_input,
        v_feedback_input,
        a_gain,
        v_gain,
        m_gain,
        a_noise,
        v_noise,
        include_noise,
        include_temporal_noise,
        a_temporal_noise,
        v_temporal_noise,
        m_temporal_noise,
    ):
        """
        Computes the temporal filtering for the neural inputs.

        Parameters
        ----------
        a_outside_input : np.ndarray
            The outside input to the auditory layer after filtering.
        v_outside_input : np.ndarray
            The outside input to the visual layer after filtering.
        m_outside_input : np.ndarray
            The outside input to the multisensory layer after filtering.
        auditoryfilter_input : np.ndarray
            The current auditory filter input.
        visualfilter_input : np.ndarray
            The current visual filter input.
        multisensoryfilter_input : np.ndarray
            The current multisensory filter input.
        t : float
            The current time in the simulation.
        a_external_input : np.ndarray
            The external input to the auditory layer neurons.
        v_external_input : np.ndarray
            The external input to the visual layer neurons.
        m_external_input : np.ndarray
            The external input to the multisensory layer neurons.
        a_cross_modal_input : np.ndarray
            The cross-modal input to the auditory layer neurons.
        v_cross_modal_input : np.ndarray
            The cross-modal input to the visual layer neurons.
        a_feedback_input : np.ndarray
            The feedback input to the auditory layer neurons.
        v_feedback_input : np.ndarray
            The feedback input to the visual layer neurons.
        a_gain : float
            The gain factor for the auditory layer.
        v_gain : float
            The gain factor for the visual layer.
        m_gain : float
            The gain factor for the multisensory layer.
        a_noise : np.ndarray
            The noise added to the auditory layer input.
        v_noise : np.ndarray
            The noise added to the visual layer input.
        include_noise : bool
            Whether to include noise in the calculations.
        include_temporal_noise : bool
            Whether to include temporal noise in the calculations.
        a_temporal_noise : float
            The temporal noise scale for the auditory layer.
        v_temporal_noise : float
            The temporal noise scale for the visual layer.
        m_temporal_noise : float
            The temporal noise scale for the multisensory layer.

        Returns
        -------
        tuple
            A tuple containing the updated outside inputs and filtered inputs
            for the auditory, visual, and multisensory layers.
        """
        if not include_noise:
            a_noise, v_noise = 0, 0

        if include_temporal_noise:
            a_tau, v_tau, m_tau = (
                a_temporal_noise,
                v_temporal_noise,
                m_temporal_noise,
            )
        else:
            a_tau, v_tau, m_tau = self.tau[0], self.tau[1], self.tau[2]

        # Auditory
        da_outside_input = auditoryfilter_input

        dauditory_filter_input = (
            (a_gain / a_tau)
            * (
                a_external_input
                + a_cross_modal_input
                + a_feedback_input
                + a_noise
            )
            - ((2 * auditoryfilter_input) / a_tau)
            - a_outside_input / np.square(a_tau)
        )

        # Visual
        dv_outside_input = visualfilter_input

        dvisual_filter_input = (
            (v_gain / v_tau)
            * (
                v_external_input
                + v_cross_modal_input
                + v_feedback_input
                + v_noise
            )
            - ((2 * visualfilter_input) / v_tau)
            - v_outside_input / np.square(v_tau)
        )

        # Multisensory
        dm_outside_input = multisensoryfilter_input

        dmultisensory_filter_input = (
            (m_gain / m_tau) * (m_external_input)
            - ((2 * multisensoryfilter_input) / m_tau)
            - m_outside_input / np.square(m_tau)
        )

        return (
            da_outside_input,
            dv_outside_input,
            dm_outside_input,
            dauditory_filter_input,
            dvisual_filter_input,
            dmultisensory_filter_input,
        )


class Paredes2025(SKNMSIMethodABC):
    r"""
    Causal Inference Network Model of Paredes et al. (2025).

    This model builds upon previous network models for multisensory integration
    (Cuppini et al., 2014; Cuppini et al., 2017) and consists of three layers:
    two unisensory layers (auditory and visual) and a multisensory layer.
    The unisensory layers encode auditory and visual stimuli separately
    and connect to the multisensory layer via feedforward
    and feedback synapses.

    The model computes implicit causal inference at the unisensory layers
    and explicit causal inference at the multisensory layer, mimicking the
    responses of neurons in the parietal-temporal association cortices.


    References
    ----------
    :cite:p:`cuppini2014neurocomputational`
    :cite:p:`cuppini2017biologically`
    :cite:p:`paredes2025excitation`


    Notes
    -----
    The Paredes2025 model maintains the neural connectivity
    (lateral, crossmodal, feedforward) and inputs as described in the
    network presented in Cuppini et al. (2017).

    This new model includes feedback connectivity and temporal filters
    as detailed below.

    The feedback synaptic weights are calculated using:

    .. math::
        B^{cm}_{jk} = B^{cm}_{0} \cdot \exp \left( -
        \frac{\left(D_{jk}\right)^{2}}{2 \left(\sigma^{cm}\right)^{2}} \right)

    where:

    - :math:`B^{cm}_{0}`: Highest level of synaptic efficacy.
    - :math:`D_{jk}`: Distance between neuron at position :math:`j` in the
      post-synaptic unisensory region and neuron at position :math:`k`
      in the pre-synaptic multisensory region.
    - :math:`\sigma^{cm}`: Width of the feedback synapses, which is the same
      for both auditory-to-multisensory (:math:`am`) and
      visual-to-multisensory (:math:`vm`) connections.

    The overall feedback input to the unisensory neurons is given by:

    .. math::
        b^{c}_{j}\left(t\right) = \sum^{N}_{k=1} B^{cm}_{jk} \cdot
        y^{c}_{k}\left(t - \Delta t_{feed}\right)

    where :math:`\Delta t_{feed}` represents the latency of
    feedback inputs between the multisensory and unisensory regions.

    The feedback synaptic weights are symmetrically defined:

    .. math::
        B_{0}^{am} = B_{0}^{vm} \quad \text{and}
                     \quad \sigma^{am} = \sigma^{vm}

    The external sources in unisensory regions are filtered using a
    second-order differential equation:

    .. math::
        \left\{
        \begin{matrix}
        \frac{d}{dt} o^{c}_{j}\left(t\right) = \delta^{c}_{j} \left(t\right) \\
        \frac{d}{dt} \delta^{c}_{j} \left(t\right) = \frac{G^{c}}{\tau^{c}}
        \cdot \left[ e^{c}_{j}\left(t\right) + c^{c}_{j}\left(t\right) +
        b^{c}_{j}\left(t\right) + n^{c}_{j} \right] - \frac{2 \cdot
        \delta^{c}_{j} \left(t\right)}{\tau^{c}} -
        \frac{o^{c}_{j}\left(t\right)}{\left( \tau^{c} \right)^{2}}
        \end{matrix}
        \right.

    where:

    - :math:`G^{c}`: Gain of the unisensory regions.
    - :math:`\tau^{c}`: Time constant of the unisensory regions.
    - :math:`c` : Indicates the unisensory region (auditory or visual).

    The external sources in multisensory regions are filtered using a
    second-order differential equation:

    .. math::
        \left\{
        \begin{matrix}
        \frac{d}{dt} o^{m}_{j}\left(t\right) = \delta^{m}_{j} \left(t\right) \\
        \frac{d}{dt} \delta^{m}_{j} \left(t\right) = \frac{G^{m}}{\tau^{m}}
        \cdot \left[ i^{m}_{j}\left(t\right) \right] -
        \frac{2 \cdot \delta^{m}_{j} \left(t\right)}{\tau^{m}}
        - \frac{o^{m}_{j}\left(t\right)}{\left( \tau^{m} \right)^{2}}
        \end{matrix}
        \right.

    where:

    - :math:`G^{m}`: Gain of the multisensory regions.
    - :math:`\tau^{m}`: Time constant of the multisensory regions.

    The cross-modal input to the unisensory neurons is calculated as:

    .. math::
        \begin{matrix}
        c^{a}_{j}\left(t\right) = \sum^{N}_{k=1} W^{av}_{jk} \cdot y^{v}_{k}
        \left(t - \Delta t_{cross} \right) \\
        c^{v}_{j}\left(t\right) = \sum^{N}_{k=1} W^{va}_{jk} \cdot y^{a}_{k}
        \left(t - \Delta t_{cross}\right)
        \end{matrix}

    where :math:`\Delta t_{cross}` represents the latency of
    cross-modal inputs between the unisensory regions.

    """

    _model_name = "Paredes2025"
    _model_type = "Neural"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_intensity", "template": "${mode0}_intensity"},
        {"target": "visual_intensity", "template": "${mode1}_intensity"},
        {"target": "auditory_duration", "template": "${mode0}_duration"},
        {"target": "visual_duration", "template": "${mode1}_duration"},
        {"target": "auditory_onset", "template": "${mode0}_onset"},
        {"target": "visual_onset", "template": "${mode1}_onset"},
    ]
    _run_output = [
        {"target": "auditory", "template": "${mode0}"},
        {"target": "visual", "template": "${mode1}"},
    ]
    _output_mode = "multi"

    def __init__(
        self,
        *,
        neurons=90,
        tau=(15, 25, 5),
        tau_neurons=1,
        s=2,
        theta=16,
        seed=None,
        mode0="auditory",
        mode1="visual",
        position_range=(0, 90),
        position_res=1,
        time_range=(0, 200),
        time_res=0.01,
        **integrator_kws,
    ):
        """
        Initialize the Paredes2025 model.

        Parameters
        ----------
        neurons : int, optional
            Number of neurons in the network. Default is 90.
        tau : tuple of float, optional
            Time constants for the temporal filters. Default is (15, 25, 5).
        tau_neurons : float, optional
            Time constant for the neuron integrator. Default is 1.
        s : float, optional
            Parameter related to the integrator model. Default is 2.
        theta : float, optional
            Threshold parameter for the integrator model. Default is 16.
        seed : int or None, optional
            Seed for the random number generator. Default is None.
        mode0 : str, optional
            The name for the first sensory modality. Default is "auditory".
        mode1 : str, optional
            The name for the second sensory modality. Default is "visual".
        position_range : tuple of int, optional
            Range of positions for stimuli in degrees as (min, max).
            Default is (0, 90).
        position_res : int or float, optional
            Resolution of the position range in degrees. Default is 1.
        time_range : tuple of float, optional
            Time range for the simulation in miliseconds. Default is (0, 200).
        time_res : float, optional
            Time resolution of the simulation in miliseconds. Default is 0.01.
        **integrator_kws
            Additional keyword arguments for the integrator.

        Raises
        ------
        ValueError
            If `tau` does not contain exactly 3 elements.
        """
        if len(tau) != 3:
            raise ValueError()

        self._neurons = neurons
        self._position_range = position_range
        self._position_res = float(position_res)
        self._time_range = time_range
        self._time_res = float(time_res)

        integrator_kws.setdefault("method", "euler")
        integrator_kws.setdefault("dt", self._time_res)

        integrator_model = Paredes2025Integrator(
            tau=tau_neurons, s=s, theta=theta
        )
        self._integrator = odeint(f=integrator_model, **integrator_kws)

        temporal_filter_model = Paredes2025TemporalFilter(tau=tau)
        self._temporal_filter = odeint(
            f=temporal_filter_model, **integrator_kws
        )

        self.set_random(np.random.default_rng(seed=seed))

        self._mode0 = mode0
        self._mode1 = mode1

    # PROPERTY ================================================================

    @property
    def neurons(self):
        """
        Returns the number of neurons in the network.

        Returns
        -------
        int
            The number of neurons.
        """
        return self._neurons

    @property
    def tau_neurons(self):
        """
        Returns the time constant for the neuron integrator.

        Returns
        -------
        float
            The time constant for the neuron integrator.
        """
        return self._integrator.f.tau

    @property
    def tau(self):
        """
        Returns the time constants for the temporal filters.

        Returns
        -------
        tuple of float
            The time constants for the temporal filters.
        """
        return self._temporal_filter.f.tau

    @property
    def s(self):
        """
        Slope of the sigmoid activation function.

        Returns
        -------
        float
            The slope parameter of the sigmoid function used in the model.
        """
        return self._integrator.f.s

    @property
    def theta(self):
        """
        Central position of the sigmoid activation function.

        Returns
        -------
        float
            The central position parameter of the sigmoid function
            used in the model.
        """
        return self._integrator.f.theta

    @property
    def random(self):
        """
        Returns the random number generator.

        Returns
        -------
        np.random.Generator
            The random number
        """
        return self._random

    @property
    def time_range(self):
        """
        Time range for simulation.

        Returns
        -------
        tuple of 2 float
            The start and end times for the simulation in seconds.
        """
        return self._time_range

    @property
    def time_res(self):
        """
        Time resolution of the simulation.

        Returns
        -------
        float
            The time step size for the simulation in seconds.
        """
        return self._time_res

    @property
    def position_range(self):
        """
        Range of positions in degrees.

        Returns
        -------
        tuple of 2 int
            The minimum and maximum positions in degrees.
        """
        return self._position_range

    @property
    def position_res(self):
        """
        Resolution of position encoding.

        Returns
        -------
        float
            The resolution of position encoding in degrees.
        """
        return self._position_res

    @property
    def mode0(self):
        """
        Returns the name of the first sensory modality.

        Returns
        -------
        str
            The name of the first sensory modality.
        """
        return self._mode0

    @property
    def mode1(self):
        """
        Returns the name of the second sensory modality.

        Returns
        -------
        str
            The name of the second sensory modality.
        """
        return self._mode1

    # Model run
    def set_random(self, rng):
        """
        Set the random number generator for the model.

        This method allows for setting a custom random number generator,
        which can be useful for ensuring reproducibility or for using
        different random number generation strategies.

        Parameters
        ----------
        rng : numpy.random.Generator
        The random number generator to be used. It should be an instance
        of `numpy.random.Generator`.
        """
        self._random = rng

    def run(
        self,
        *,
        auditory_soa=50,
        visual_soa=None,
        auditory_onset=16,
        visual_onset=16,
        auditory_duration=7,
        visual_duration=12,
        auditory_position=None,
        visual_position=None,
        auditory_intensity=2.4,
        visual_intensity=1.4,
        auditory_sigma=32,
        visual_sigma=4,
        noise=False,
        noise_level=0.40,
        temporal_noise=False,
        temporal_noise_scale=5,
        lateral_excitation=2,
        lateral_excitation_sigma=3,
        lateral_inhibition=1.8,
        lateral_inhibition_sigma=24,
        cross_modal_weight=0.075,
        cross_modal_latency=16,
        feed_latency=95,
        feedback_weight=0.10,
        feedforward_weight=1.4,
        auditory_gain=None,
        visual_gain=None,
        multisensory_gain=None,
        auditory_stim_n=2,
        visual_stim_n=1,
        feedforward_pruning_threshold=0,
        cross_modal_pruning_threshold=0,
        causes_kind="count",
        causes_dim="space",
        causes_peak_threshold=0.80,
        causes_peak_distance=None,
    ):
        """
        Runs the model simulation with specified parameters.

        Parameters
        ----------
        auditory_soa : float, optional
            Stimulus-onset asynchrony for auditory stimuli (default is 50).
        visual_soa : float, optional
            Stimulus-onset asynchrony for visual stimuli (default is None).
        auditory_onset : float, optional
            Onset time for auditory stimuli (default is 16).
        visual_onset : float, optional
            Onset time for visual stimuli (default is 16).
        auditory_duration : float, optional
            Duration of auditory stimuli (default is 7).
        visual_duration : float, optional
            Duration of visual stimuli (default is 12).
        auditory_position : float, optional
            Position of auditory stimuli (default is middle of the range).
        visual_position : float, optional
            Position of visual stimuli (default is middle of the range).
        auditory_intensity : float, optional
            Intensity of auditory stimuli (default is 2.4).
        visual_intensity : float, optional
            Intensity of visual stimuli (default is 1.4).
        auditory_sigma : float, optional
            Standard deviation for auditory stimuli (default is 32).
        visual_sigma : float, optional
            Standard deviation for visual stimuli (default is 4).
        noise : bool, optional
            Whether to include noise in the simulation (default is False).
        noise_level : float, optional
            Level of noise to add (default is 0.40).
        temporal_noise : bool, optional
            Whether to include temporal noise (default is False).
        temporal_noise_scale : float, optional
            Scale of temporal noise (default is 5).
        lateral_excitation : float, optional
            Lateral excitation weight parameter (default is 2).
        lateral_excitation_sigma : float, optional
            Lateral excitation spread parameter (default is 3).
        lateral_inhibition : float, optional
            Lateral inhibition weight parameter (default is 1.8).
        lateral_inhibition_sigma : float, optional
            Lateral inhibition spread parameter (default is 24).
        cross_modal_weight : float, optional
            Weight for cross-modal connections (default is 0.075).
        cross_modal_latency : float, optional
            Latency for cross-modal inputs (default is 16).
        feed_latency : float, optional
            Latency for feedforward inputs (default is 95).
        feedback_weight : float, optional
            Weight for feedback connections (default is 0.10).
        feedforward_weight : float, optional
            Weight for feedforward connections (default is 1.4).
        auditory_gain : float, optional
            Gain for auditory processing
            (default is None, which sets to exp(1)).
        visual_gain : float, optional
            Gain for visual processing
            (default is None, which sets to exp(1)).
        multisensory_gain : float, optional
            Gain for multisensory processing
            (default is None, which sets to exp(1)).
        auditory_stim_n : int, optional
            Number of auditory stimuli (default is 2).
        visual_stim_n : int, optional
            Number of visual stimuli (default is 1).
        feedforward_pruning_threshold : float, optional
            Threshold for pruning feedforward synapses (default is 0).
        cross_modal_pruning_threshold : float, optional
            Threshold for pruning cross-modal synapses (default is 0).
        causes_kind : str, optional
            Method for calculating causes ("count" or other)
            (default is "count").
        causes_dim : str, optional
            Dimension for calculating causes ("space" or other)
            (default is "space").
        causes_peak_threshold : float, optional
            Peak threshold for causes calculation (default is 0.80).

        Returns
        -------
        tuple
            A tuple containing:
            - response (dict): A dictionary with keys "auditory", "visual",
            and "multi", containing the simulation results for each layer.
            - extra (dict): A dictionary with additional information such as
            total inputs, causes parameters, and stimulus positions.
        """
        auditory_position = (
            int(self._position_range[1] / 2)
            if auditory_position is None
            else auditory_position
        )

        visual_position = (
            int(self._position_range[1] / 2)
            if visual_position is None
            else visual_position
        )

        auditory_gain = np.exp(1) if auditory_gain is None else auditory_gain
        visual_gain = np.exp(1) if visual_gain is None else visual_gain
        multisensory_gain = (
            np.exp(1) if multisensory_gain is None else multisensory_gain
        )

        hist_times = np.arange(
            self._time_range[0], self._time_range[1], self._integrator.dt
        )

        sim_cross_modal_latency = int(
            cross_modal_latency / self._integrator.dt
        )

        sim_feed_latency = int(feed_latency / self._integrator.dt)

        # Build synapses
        auditory_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=lateral_excitation_sigma,
            inhibition_scale=lateral_inhibition_sigma,
        )
        visual_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=lateral_excitation_sigma,
            inhibition_scale=lateral_inhibition_sigma,
        )
        multi_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=lateral_excitation_sigma,
            inhibition_scale=lateral_inhibition_sigma,
        )
        auditory_to_visual_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=cross_modal_weight, sigma=5
        )
        visual_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=cross_modal_weight, sigma=5
        )
        auditory_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=feedforward_weight, sigma=0.5
        )
        visual_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=feedforward_weight, sigma=0.5
        )
        multi_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=feedback_weight, sigma=0.5
        )
        multi_to_visual_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=feedback_weight, sigma=0.5
        )

        # Prune synapses
        auditory_to_multi_synapses = prune_synapses(
            auditory_to_multi_synapses, feedforward_pruning_threshold
        )
        visual_to_multi_synapses = prune_synapses(
            visual_to_multi_synapses, feedforward_pruning_threshold
        )
        auditory_to_visual_synapses = prune_synapses(
            auditory_to_visual_synapses, cross_modal_pruning_threshold
        )
        visual_to_auditory_synapses = prune_synapses(
            visual_to_auditory_synapses, cross_modal_pruning_threshold
        )

        # Generate Stimuli
        point_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_intensity,
            scale=auditory_sigma,
            loc=auditory_position,
        )
        point_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_intensity,
            scale=visual_sigma,
            loc=visual_position,
        )

        auditory_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_auditory_stimuli,
            stimuli_duration=auditory_duration,
            onset=auditory_onset,
            simulation_length=self._time_range[1],
            time_res=self.time_res,
            dt=self._integrator.dt,
            stimuli_n=auditory_stim_n,
            soa=auditory_soa,
        )

        visual_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=visual_onset,
            simulation_length=self._time_range[1],
            time_res=self.time_res,
            dt=self._integrator.dt,
            stimuli_n=visual_stim_n,
            soa=visual_soa,
        )

        # Data holders
        z_1d = np.zeros(self.neurons)
        auditory_y, visual_y, multi_y = (
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
        )
        (
            auditory_outside_input,
            visual_outside_input,
            multisensory_outside_input,
        ) = (copy.deepcopy(z_1d), copy.deepcopy(z_1d), copy.deepcopy(z_1d))
        auditoryfilter_input, visualfilter_input, multisensoryfilter_input = (
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
        )

        z_2d = np.zeros(
            (int(self._time_range[1] / self._integrator.dt), self.neurons)
        )
        auditory_res, visual_res, multi_res = (
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
        )

        (
            auditory_total_inputs,
            visual_total_inputs,
            multisensory_total_inputs,
        ) = (copy.deepcopy(z_2d), copy.deepcopy(z_2d), copy.deepcopy(z_2d))

        del z_1d, z_2d

        # Temporal noise
        rand_a_tau, rand_v_tau, rand_m_tau = (
            self.random.uniform(
                self.tau[0] - temporal_noise_scale / 2,
                self.tau[0] + temporal_noise_scale / 2,
            ),
            self.random.uniform(
                self.tau[1] - temporal_noise_scale / 2,
                self.tau[1] + temporal_noise_scale / 2,
            ),
            self.random.uniform(
                self.tau[2] - temporal_noise_scale / 2,
                self.tau[2] + temporal_noise_scale / 2,
            ),
        )

        for i in range(hist_times.size):
            time = int(hist_times[i] / self._integrator.dt)

            # Input noise
            auditory_noise = -(auditory_intensity * noise_level) + (
                2 * auditory_intensity * noise_level
            ) * self.random.random(self.neurons)
            visual_noise = -(visual_intensity * noise_level) + (
                2 * visual_intensity * noise_level
            ) * self.random.random(self.neurons)

            # Compute cross-modal input
            computed_cross_latency = compute_latency(
                time, sim_cross_modal_latency
            )

            auditory_cm_input = np.sum(
                visual_to_auditory_synapses.T
                * visual_res[computed_cross_latency, :],
                axis=1,
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses.T
                * auditory_res[computed_cross_latency, :],
                axis=1,
            )

            # Compute feedback input
            computed_feed_latency = compute_latency(time, sim_feed_latency)

            auditory_feedback_input = np.sum(
                multi_to_auditory_synapses.T
                * multi_res[computed_feed_latency, :],
                axis=1,
            )
            visual_feedback_input = np.sum(
                multi_to_visual_synapses.T
                * multi_res[computed_feed_latency, :],
                axis=1,
            )

            # Compute feedforward input
            multi_input = np.sum(
                auditory_to_multi_synapses.T
                * auditory_res[computed_feed_latency, :],
                axis=1,
            ) + np.sum(
                visual_to_multi_synapses.T
                * visual_res[computed_feed_latency, :],
                axis=1,
            )

            (
                auditory_outside_input,
                visual_outside_input,
                multisensory_outside_input,
                auditoryfilter_input,
                visualfilter_input,
                multisensoryfilter_input,
            ) = self._temporal_filter(
                a_outside_input=auditory_outside_input,
                v_outside_input=visual_outside_input,
                m_outside_input=multisensory_outside_input,
                auditoryfilter_input=auditoryfilter_input,
                visualfilter_input=visualfilter_input,
                multisensoryfilter_input=multisensoryfilter_input,
                t=time,
                a_external_input=auditory_stimuli[i],
                v_external_input=visual_stimuli[i],
                m_external_input=multi_input,
                a_cross_modal_input=auditory_cm_input,
                v_cross_modal_input=visual_cm_input,
                a_feedback_input=auditory_feedback_input,
                v_feedback_input=visual_feedback_input,
                a_gain=auditory_gain,
                v_gain=visual_gain,
                m_gain=multisensory_gain,
                a_noise=auditory_noise,
                v_noise=visual_noise,
                include_noise=noise,
                include_temporal_noise=temporal_noise,
                a_temporal_noise=rand_a_tau,
                v_temporal_noise=rand_v_tau,
                m_temporal_noise=rand_m_tau,
            )

            # Compute lateral inpunt
            la = np.sum(auditory_latsynapses.T * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses.T * visual_y, axis=1)
            lm = np.sum(multi_latsynapses.T * multi_y, axis=1)

            # Compute unisensory total input
            auditory_u = la + auditory_outside_input
            visual_u = lv + visual_outside_input

            # Compute multisensory total input
            u_m = lm + multisensory_outside_input

            (
                auditory_total_inputs[i, :],
                visual_total_inputs[i, :],
                multisensory_total_inputs[i, :],
            ) = (
                auditory_u,
                visual_u,
                u_m,
            )

            # Compute neurons activity
            auditory_y, visual_y, multi_y = self._integrator(
                y_a=auditory_y,
                y_v=visual_y,
                y_m=multi_y,
                t=time,
                u_a=auditory_u,
                u_v=visual_u,
                u_m=u_m,
            )

            auditory_res[i, :], visual_res[i, :], multi_res[i, :] = (
                auditory_y,
                visual_y,
                multi_y,
            )

        response = {
            "auditory": auditory_res,
            "visual": visual_res,
            "multi": multi_res,
        }

        extra = {
            "auditory_total_input": auditory_total_inputs,
            "visual_total_input": visual_total_inputs,
            "multi_total_input": multisensory_total_inputs,
            "causes_kind": causes_kind,
            "causes_dim": causes_dim,
            "causes_peak_threshold": causes_peak_threshold,
            "causes_peak_distance": causes_peak_distance,
            "stim_position": [auditory_position, visual_position],
        }

        return response, extra

    def calculate_causes(
        self,
        multi,
        causes_kind,
        causes_dim,
        causes_peak_threshold,
        causes_peak_distance,
        stim_position,
        **kwargs,
    ):
        """
        Calculate the causes based on spatiotemporal peaks.

        This method computes the causes (i.e., the underlying factors
        or sources) of multisensory activity based on the peaks in the
        multisensory data. The calculation considers the specified method and
        dimension for cause determination.

        Parameters
        ----------
        multi : np.ndarray
            Multisensory activity data.
        causes_kind : str
            Method for calculating causes ("count" or other).
        causes_dim : str
            Dimension for calculating causes ("space" or other).
        causes_peak_threshold : float
            Peak threshold for causes calculation.
        stim_position : list of float
            List containing the positions of the stimuli.
        **kwargs : keyword arguments
            Additional arguments for the causes calculation.

        Returns
        -------
        causes : np.ndarray
            Calculated causes based on the specified method and parameters.
        """
        # Calculate the average stimuli position
        position = int(np.mean([stim_position[0], stim_position[1]]))

        # Calculates the causes in the desired dimension
        # using the specified method
        causes = calculate_spatiotemporal_causes_from_peaks(
            mode_spatiotemporal_activity_data=multi,
            causes_kind=causes_kind,
            causes_dim=causes_dim,
            peak_threshold=causes_peak_threshold,
            peak_distance=causes_peak_distance,
            time_point=-1,
            spatial_point=position,
        )

        # Return the calculated causes
        return causes
