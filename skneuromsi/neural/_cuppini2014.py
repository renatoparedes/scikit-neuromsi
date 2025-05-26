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
    calculate_lateral_synapses,
    calculate_stimuli_input,
    compute_latency,
    create_unimodal_stimuli_matrix,
)


@dataclass
class Cuppini2014Integrator:
    """
    Integrator function for the Cuppini2014 model.

    This class represents the integrator function used to compute
    the dynamics of the neural network according to the Cuppini (2014) model.
    It handles the update rules for the neurons' activities based on
    their net inputs.

    Parameters
    ----------
    tau : tuple of 3 float
        Time constants for the auditory, visual,
        and multisensory neurons, respectively.
    s : float
        Slope of the sigmoid activation function.
    theta : float
        Central position of the sigmoid activation function.

    """

    #: Time constants for the auditory, visual, and multisensory neurons
    tau: tuple

    #: Slope of the sigmoid activation function
    s: float

    #: Central position of the sigmoid activation function
    theta: float

    #: Name of the integrator
    name: str = "Cuppini2014Integrator"

    @property
    def __name__(self):
        """Return the name of the Integrator."""
        return self.name

    def sigmoid(self, u):
        """Computes the sigmoid activation function."""
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def __call__(self, y_a, y_v, t, u_a, u_v):
        """Computes the activites of neurons."""
        # Auditory
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau[0])

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau[0])

        return dy_a, dy_v


@dataclass
class Cuppini2014TemporalFilter:
    """Temporal filter for the Cuppini2014 model."""

    tau: tuple
    name: str = "Cuppini2014TemporalFilter"

    @property
    def __name__(self):
        """Return the name of the Temporal Filter."""
        return self.name

    def __call__(
        self,
        a_outside_input,
        v_outside_input,
        auditoryfilter_input,
        visualfilter_input,
        t,
        a_external_input,
        v_external_input,
        a_cross_modal_input,
        v_cross_modal_input,
        a_gain,
        v_gain,
    ):
        """
        Computes the temporal filtering for the neural inputs.

        Parameters
        ----------
        a_outside_input : np.ndarray
            The outside input to the auditory layer after filtering.
        v_outside_input : np.ndarray
            The outside input to the visual layer after filtering.
        auditoryfilter_input : np.ndarray
            The current auditory filter input.
        visualfilter_input : np.ndarray
            The current visual filter input.
        t : float
            The current time in the simulation.
        a_external_input : np.ndarray
            The external input to the auditory layer neurons.
        v_external_input : np.ndarray
            The external input to the visual layer neurons.
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

        Returns
        -------
        tuple
            A tuple containing the updated outside inputs and filtered inputs
            for the auditory and visual layers.
        """
        # Auditory

        da_outside_input = auditoryfilter_input

        dauditory_filter_input = (
            (a_gain / self.tau[1]) * (a_external_input + a_cross_modal_input)
            - ((2 * auditoryfilter_input) / self.tau[1])
            - a_outside_input / np.square(self.tau[1])
        )

        # Visual

        dv_outside_input = visualfilter_input

        dvisual_filter_input = (
            (v_gain / self.tau[2]) * (v_external_input + v_cross_modal_input)
            - ((2 * visualfilter_input) / self.tau[2])
            - v_outside_input / np.square(self.tau[2])
        )

        return (
            da_outside_input,
            dv_outside_input,
            dauditory_filter_input,
            dvisual_filter_input,
        )


class Cuppini2014(SKNMSIMethodABC):
    """Network model for multisensory integration of Cuppini et al. (2014).

    This model simulates neural responses in a multisensory system,
    consisting of auditory and visual areas. By default, each of these areas
    consists of 180 neurons arranged topologically to encode a 180° space.
    In this way, each neuron encodes 1° of space and neurons close to each
    other encode close spatial positions. The model includes a temporal filter
    to accurately reproduce temporal dynamics of multisensory integration.


    References
    ----------
    :cite:p:`cuppini2014neurocomputational`

    """

    _model_name = "Cuppini2014"
    _model_type = "Neural"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_intensity", "template": "${mode0}_intensity"},
        {"target": "visual_intensity", "template": "${mode1}_intensity"},
        {"target": "auditory_duration", "template": "${mode0}_duration"},
        {"target": "visual_duration", "template": "${mode1}_duration"},
        {"target": "auditory_gain", "template": "${mode0}_gain"},
        {"target": "visual_gain", "template": "${mode1}_gain"},
    ]
    _run_output = [
        {"target": "auditory", "template": "${mode0}"},
        {"target": "visual", "template": "${mode1}"},
    ]
    _output_mode = "multi"

    def __init__(
        self,
        *,
        neurons=180,
        tau=(1, 15, 25),  # neuron, auditory and visual
        s=2,
        theta=16,
        seed=None,
        mode0="auditory",
        mode1="visual",
        position_range=(0, 180),
        position_res=1,
        time_range=(0, 100),
        time_res=0.01,
        **integrator_kws,
    ):
        """
        Initializes the Cuppini2014 model.

        Parameters
        ----------
        neurons : int, optional
            Number of neurons per layer. Default is 180.
        tau : tuple of 3 float, optional
            Time constants for the auditory, visual, and multisensory neurons,
            respectively. Default is (3, 15, 1).
        s : float, optional
            Slope of the sigmoid activation function. Default is 0.3.
        theta : float, optional
            Central position of the sigmoid activation function. Default is 20.
        seed : int or None, optional
            Seed for the random number generator.
            If None, the random number generator will not be seeded.
            Default is None.
        mode0 : str, optional
            The name for the first sensory modality. Default is "auditory".
        mode1 : str, optional
            The name for the second sensory modality. Default is "visual".
        position_range : tuple of 2 int, optional
            Range of positions in degrees as (min, max). Default is (0, 180).
        position_res : float, optional
            Resolution of positions in degrees. Default is 1.
        time_range : tuple of 2 float, optional
            Time range for the simulation as (start, end) in miliseconds.
            Default is (0, 100).
        time_res : float, optional
            Time resolution for the simulation in miliseconds. Default is 0.01.
        **integrator_kws
            Additional keyword arguments passed to the integrator.
            These can include parameters such as the integration method and
            time step size.

        Raises
        ------
        ValueError
            If the length of `tau` is not equal to 3.

        Attributes
        ----------
        neurons : int
            Number of neurons per layer.
        tau : tuple of 3 float
            Time constants for the auditory, visual, and multisensory neurons.
        s : float
            Slope of the sigmoid activation function.
        theta : float
            Central position of the sigmoid activation function.
        random : np.random.Generator
            Random number generator.
        time_range : tuple of 2 float
            Time range for the simulation.
        time_res : float
            Time resolution for the simulation.
        position_range : tuple of 2 int
            Range of positions in degrees.
        position_res : float
            Resolution of positions in degrees.
        mode0 : str
            Name of the auditory modality.
        mode1 : str
            Name of the visual modality.
        dtype : np.dtype
            Data type used for computations.
        _integrator_function : Cuppini2014IntegratorFunction
            The integrator function used for simulation.
        _integrator_kws : dict
            Keyword arguments for the integrator.
        _integrator : callable
            The integrator function.
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

        integrator_model = Cuppini2014Integrator(tau=tau, s=s, theta=theta)
        self._integrator = odeint(f=integrator_model, **integrator_kws)

        temporal_filter_model = Cuppini2014TemporalFilter(tau=tau)
        self._temporal_filter = odeint(
            f=temporal_filter_model, **integrator_kws
        )

        self._mode0 = mode0
        self._mode1 = mode1

        self.set_random(np.random.default_rng(seed=seed))

    # PROPERTY ================================================================

    @property
    def neurons(self):
        """
        Number of neurons in each layer.

        Returns
        -------
        int
            The number of neurons used in the simulation.
        """
        return self._neurons

    @property
    def tau(self):
        """
        Time constants for the neurons.

        Returns
        -------
        tuple of 3 float
            Time constants for the auditory, visual, and
            multisensory neurons, respectively.
        """
        return self._integrator.f.tau

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
        Random number generator.

        Returns
        -------
        numpy.random.Generator
            The random number generator used for initialization.
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
        which can be useful for ensuring reproducibility or
        for using different random number generation strategies.

        Parameters
        ----------
        rng : numpy.random.Generator
        The random number generator to be used.
        It should be an instance of `numpy.random.Generator`.
        """
        self._random = rng

    def run(
        self,
        *,
        soa=56,
        onset=25,
        auditory_duration=10,
        visual_duration=20,
        auditory_position=None,
        visual_position=None,
        auditory_intensity=3,
        visual_intensity=1,
        noise=False,
        lateral_excitation=2,
        lateral_inhibition=1.8,
        cross_modal_latency=16,
        auditory_gain=None,
        visual_gain=None,
        auditory_stim_n=2,
        visual_stim_n=1,
    ):
        """
        Run the simulation of the Cuppini2014 model.

        Parameters
        ----------
        soa : float, optional
            Stimulus-onset asynchrony for stimuli (default is 50).
        onset : float, optional
            Onset time for stimuli (default is 16).
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
        noise : bool, optional
            Whether to include noise in the simulation (default is False).
        lateral_excitation : float, optional
            Lateral excitation parameter (default is 2).
        lateral_inhibition : float, optional
            Lateral inhibition parameter (default is 1.8).
        cross_modal_latency : float, optional
            Latency for cross-modal inputs (default is 16).
        auditory_gain : float, optional
            Gain for auditory processing
            (default is None, which sets to exp(1)).
        visual_gain : float, optional
            Gain for visual processing
            (default is None, which sets to exp(1)).
        auditory_stim_n : int, optional
            Number of auditory stimuli (default is 2).
        visual_stim_n : int, optional
            Number of visual stimuli (default is 1).

        Returns
        -------
        tuple
            A tuple containing:
            - response (dict): A dictionary with keys "auditory", "visual",
            and "multi", containing the simulation results for each layer.
            - extra (dict): A dictionary with additional information such as
            inputs and filter inputs.
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

        hist_times = np.arange(
            self._time_range[0], self._time_range[1], self._integrator.dt
        )

        sim_cross_modal_latency = int(
            cross_modal_latency / self._integrator.dt
        )

        # Build synapses
        auditory_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=24,
        )
        visual_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=24,
        )

        cross_modal_synapses_weight = 0.35

        # Generate Stimuli
        point_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_intensity,
            scale=32,
            loc=auditory_position,
        )
        point_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_intensity,
            scale=4,
            loc=visual_position,
        )

        auditory_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_auditory_stimuli,
            stimuli_duration=auditory_duration,
            onset=onset,
            simulation_length=self._time_range[1],
            time_res=self.time_res,
            dt=self._integrator.dt,
            stimuli_n=auditory_stim_n,
            soa=soa,
        )

        visual_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=onset,
            simulation_length=self._time_range[1],
            time_res=self.time_res,
            dt=self._integrator.dt,
            stimuli_n=visual_stim_n,
        )

        # Data holders
        z_1d = np.zeros(self.neurons)
        auditory_y, visual_y = copy.deepcopy(z_1d), copy.deepcopy(z_1d)
        auditory_outside_input, visual_outside_input = copy.deepcopy(
            z_1d
        ), copy.deepcopy(z_1d)
        auditoryfilter_input, visualfilter_input = copy.deepcopy(
            z_1d
        ), copy.deepcopy(z_1d)

        # template for the next holders
        z_2d = np.zeros(
            (int(self._time_range[1] / self._integrator.dt), self.neurons)
        )

        auditory_res, visual_res, multi_res = (
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
        )
        auditory_outside_inputs, visual_outside_inputs = copy.deepcopy(
            z_2d
        ), copy.deepcopy(z_2d)
        auditoryfilter_inputs, visualfilter_inputs = copy.deepcopy(
            z_2d
        ), copy.deepcopy(z_2d)
        auditory_lateral_inputs, visual_lateral_inputs = copy.deepcopy(
            z_2d
        ), copy.deepcopy(z_2d)
        auditory_total_inputs, visual_total_inputs = copy.deepcopy(
            z_2d
        ), copy.deepcopy(z_2d)

        del z_1d, z_2d

        for i in range(hist_times.size):
            time = int(hist_times[i] / self._integrator.dt)

            # Compute cross-modal input
            computed_cross_latency = compute_latency(
                time, sim_cross_modal_latency
            )

            auditory_cm_input = (
                cross_modal_synapses_weight
                * visual_res[computed_cross_latency, :]
            )
            visual_cm_input = (
                cross_modal_synapses_weight
                * auditory_res[computed_cross_latency, :]
            )

            (
                auditory_outside_input,
                visual_outside_input,
                auditoryfilter_input,
                visualfilter_input,
            ) = self._temporal_filter(
                a_outside_input=auditory_outside_input,
                v_outside_input=visual_outside_input,
                auditoryfilter_input=auditoryfilter_input,
                visualfilter_input=visualfilter_input,
                t=time,
                a_external_input=auditory_stimuli[i],
                v_external_input=visual_stimuli[i],
                a_cross_modal_input=auditory_cm_input,
                v_cross_modal_input=visual_cm_input,
                a_gain=auditory_gain,
                v_gain=visual_gain,
            )

            auditory_outside_inputs[i, :], visual_outside_inputs[i, :] = (
                auditory_outside_input,
                visual_outside_input,
            )

            auditoryfilter_inputs[i, :], visualfilter_inputs[i, :] = (
                auditoryfilter_input,
                visualfilter_input,
            )

            # Compute lateral input
            la = np.sum(auditory_latsynapses.T * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses.T * visual_y, axis=1)

            auditory_lateral_inputs[i, :], visual_lateral_inputs[i, :] = la, lv

            # Compute unisensory total input
            auditory_u = la + auditory_outside_input
            visual_u = lv + visual_outside_input

            auditory_total_inputs[i, :], visual_total_inputs[i, :] = (
                auditory_u,
                visual_u,
            )

            # Compute neurons activity
            auditory_y, visual_y = self._integrator(
                y_a=auditory_y,
                y_v=visual_y,
                t=time,
                u_a=auditory_u,
                u_v=visual_u,
            )

            auditory_res[i, :], visual_res[i, :] = (
                auditory_y,
                visual_y,
            )

        response = {
            "auditory": auditory_res,
            "visual": visual_res,
            "multi": multi_res,
        }

        return response, {
            "auditory_stimuli": auditory_stimuli,
            "visual_stimuli": visual_stimuli,
            "auditory_outside_input": auditory_outside_inputs,
            "visual_outside_input": visual_outside_inputs,
            "auditory_lateral_input": auditory_lateral_inputs,
            "visual_lateral_input": visual_lateral_inputs,
            "auditory_total_input": auditory_total_inputs,
            "visual_total_input": visual_total_inputs,
            "auditory_lateral_synapses": auditory_latsynapses,
            "visual_lateral_synapses": visual_latsynapses,
            "auditory_filter_input": auditoryfilter_inputs,
            "visual_filter_input": visualfilter_inputs,
        }
