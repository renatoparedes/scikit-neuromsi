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
    calculate_rf,
    calculate_w,
    calculate_stimuli_input,
    compute_latency,
    create_unimodal_stimuli_matrix,
    prune_synapses,
)
from ..utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


@dataclass
class Paredes2027Integrator:
    """A class representing the integrator for the Paredes2027 model."""

    tau: tuple
    tau_e: tuple
    tau_s: tuple
    s: float
    theta: float
    nu: float
    c: float
    c_t: float
    w: np.array
    name: str = "Paredes2027Integrator"

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

    def __call__(self, y_a, y_v, y_m, e_a, e_v, e_m, s_a, s_v, s_m, t, u_a, u_v, u_m):
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

        a_tau, v_tau, m_tau = self.tau[0], self.tau[1], self.tau[2]
        a_tau_e, v_tau_e, m_tau_e = self.tau_e[0], self.tau_e[1], self.tau_e[2]
        a_tau_s, v_tau_s, m_tau_s = self.tau_s[0], self.tau_s[1], self.tau_s[2]

        # Apply sigmoid to neural input
        I_a = self.sigmoid(u_a)
        I_v = self.sigmoid(u_v)
        I_m = self.sigmoid(u_m)

        # Excitatory drive
        de_a = (-e_a + I_a) * (1 / a_tau_e)
        de_v = (-e_v + I_v) * (1 / v_tau_e)
        de_m = (-e_m + I_m) * (1 / m_tau_e)

        # Supressive drive
        ds_a = (-s_a + e_a) * (1 / a_tau_s)
        ds_v = (-s_v + e_v) * (1 / v_tau_s)
        ds_m = (-s_m + e_m) * (1 / m_tau_s)

        # Calculate suppresive population pools
        pool_a_suppressive = self.w @ s_a
        pool_v_suppressive = self.w @ s_v
        pool_m_suppressive = self.w @ s_m

        # Calculate auditory activity
        normalized_a = e_a / (self.nu + self.c * pool_a_suppressive)
        dy_a = (-y_a + normalized_a) * (1 / a_tau)

        # Calculate visual activity
        normalized_v = e_v / (self.nu + self.c * pool_v_suppressive)
        dy_v = (-y_v + normalized_v) * (1 / v_tau)

        # Calculate multisensory activity
        normalized_m = e_m / (self.nu + self.c * pool_m_suppressive)
        dy_m = (-y_m + normalized_m) * (1 / m_tau)

        return dy_a, dy_v, dy_m, de_a, de_v, de_m, ds_a, ds_v, ds_m


class Paredes2027_DSTN(SKNMSIMethodABC):
    r"""
    Causal Inference Network Model of Paredes et al. (2027).

    """

    _model_name = "Paredes2027_DSTN"
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
        {"target": "auditory_stim_sigma", "template": "${mode0}_stim_sigma"},
        {"target": "visual_stim_sigma", "template": "${mode1}_stim_sigma"},
        {"target": "auditory_rf_sigma", "template": "${mode0}_rf_sigma"},
        {"target": "visual_rf_sigma", "template": "${mode1}_rf_sigma"},
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
        tau=(30, 50, 6),
        tau_e=(50, 300, 150),
        tau_s=(30, 50, 50),
        s=2,
        theta=16,
        nu=1,
        c=0.5,
        c_t=0.5,
        w_sigma=3.0,
        seed=None,
        mode0="auditory",
        mode1="visual",
        position_range=(0, 90),
        position_res=1,
        time_range=(0, 300),
        time_res=0.01,
        **integrator_kws,
    ):
        """
        Initialize the Paredes2027 DSTN model.

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

        w = calculate_w(neurons=self.neurons, w_sigma=w_sigma)
        
        integrator_model = Paredes2027Integrator(
            tau=tau, tau_e=tau_e, tau_s=tau_s, s=s, theta=theta, c=c, c_t=c_t, nu=nu, w=w,
        )
        self._integrator = odeint(f=integrator_model, **integrator_kws)

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
        auditory_stim_sigma=2.5,
        visual_stim_sigma=2.5,
        auditory_rf_sigma=20,
        visual_rf_sigma=4,
        noise=False,
        noise_level=0.40,
        lateral_excitation=2,
        lateral_excitation_sigma=3,
        lateral_inhibition=1.8,
        lateral_inhibition_sigma=24,
        cross_modal_weight=0.075,
        cross_modal_sigma=5,
        cross_modal_latency=16,
        feed_latency=50,
        feedback_weight=0.10,
        feedforward_weight=1.4,
        auditory_stim_n=2,
        visual_stim_n=1,
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
        auditory_stim_sigma : float, optional
            Standard deviation for auditory stimuli (default is 32).
        visual_stim_sigma : float, optional
            Standard deviation for visual stimuli (default is 4).
        noise : bool, optional
            Whether to include noise in the simulation (default is False).
        noise_level : float, optional
            Level of noise to add (default is 0.40).
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
        auditory_stim_n : int, optional
            Number of auditory stimuli (default is 2).
        visual_stim_n : int, optional
            Number of visual stimuli (default is 1).
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
            neurons=self.neurons, weight=cross_modal_weight, sigma=cross_modal_sigma,
        )
        visual_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=cross_modal_weight, sigma=cross_modal_sigma,
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

        # Generate Physical Stimuli
        raw_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_intensity,
            scale=auditory_stim_sigma,
            loc=auditory_position,
        )
        raw_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_intensity,
            scale=visual_stim_sigma,
            loc=visual_position,
        )

        # Generate RF
        auditory_rf = calculate_rf(neurons=self.neurons,
                                   rf_sigma=auditory_rf_sigma)

        visual_rf = calculate_rf(neurons=self.neurons,
                                           rf_sigma=visual_rf_sigma)
        
        # Convolve Stimului with RF
        point_auditory_stimuli = auditory_rf @ raw_auditory_stimuli
        point_visual_stimuli = visual_rf @ raw_visual_stimuli

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
        auditory_e, visual_e, multi_e = (
            copy.deepcopy(z_1d), copy.deepcopy(z_1d), copy.deepcopy(z_1d),
        )
        auditory_s, visual_s, multi_s = (
            copy.deepcopy(z_1d), copy.deepcopy(z_1d), copy.deepcopy(z_1d),
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

            # Compute outside input
            auditory_outside_input = auditory_stimuli[i] + auditory_cm_input + auditory_feedback_input
            visual_outside_input = visual_stimuli[i] + visual_cm_input + visual_feedback_input

            # Compute lateral inpunt
            la = np.sum(auditory_latsynapses.T * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses.T * visual_y, axis=1)
            lm = np.sum(multi_latsynapses.T * multi_y, axis=1)

            # Compute unisensory total input
            auditory_u = la + auditory_outside_input
            visual_u = lv + visual_outside_input

            # Compute multisensory total input
            multi_u = lm + multi_input

            (
                auditory_total_inputs[i, :],
                visual_total_inputs[i, :],
                multisensory_total_inputs[i, :],
            ) = (
                auditory_u,
                visual_u,
                multi_u,
            )

            # Compute neurons activity
            (auditory_y, visual_y, multi_y, 
             auditory_e, visual_e, multi_e, 
             auditory_s, visual_s, multi_s) = self._integrator(
                y_a=auditory_y, y_v=visual_y, y_m=multi_y,
                e_a=auditory_e, e_v=visual_e, e_m=multi_e,
                s_a=auditory_s, s_v=visual_s, s_m=multi_s,
                t=time,
                u_a=auditory_u, u_v=visual_u, u_m=multi_u,
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
