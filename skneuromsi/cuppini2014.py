#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


from dataclasses import dataclass

import brainpy as bp

import numpy as np

from .core import SKNMSIMethodABC, MDResult


@dataclass
class Cuppini2014Integrator:
    tau: tuple
    s: float
    theta: float
    name: str = "Cuppini2014Integrator"

    @property
    def __name__(self):
        return self.name

    def sigmoid(self, u):
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def __call__(self, y_a, y_v, t, u_a, u_v):

        # Auditory
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau[0])

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau[0])

        return dy_a, dy_v


@dataclass
class Cuppini2014TemporalFilter:
    tau: tuple
    name: str = "Cuppini2014TemporalFilter"

    @property
    def __name__(self):
        return self.name

    def __call__(
        self,
        a_outside_input,
        v_outside_input,
        t,
        a_external_input,
        v_external_input,
        a_cross_modal_input,
        v_cross_modal_input,
        a_gain,
        v_gain,
    ):

        # Auditory
        dauditory_outside_input = (
            a_gain / self.tau[1] * (a_external_input + a_cross_modal_input)
            - ((2 * a_outside_input) / self.tau[1])
            - a_outside_input / np.square(self.tau[1])
        )

        # Visual
        dvisual_outside_input = (
            v_gain / self.tau[2] * (v_external_input + v_cross_modal_input)
            - ((2 * v_outside_input) / self.tau[2])
            - v_outside_input / np.square(self.tau[2])
        )

        return dauditory_outside_input, dvisual_outside_input


class Cuppini2014(SKNMSIMethodABC):
    """Zaraza.


    References
    ----------
    :cite:p:`cuppini2017biologically`

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
    _run_result = MDResult

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
        **integrator_kws,
    ):
        if len(tau) != 3:
            raise ValueError()

        self._neurons = neurons
        self._random = np.random.default_rng(seed=seed)

        integrator_kws.setdefault("method", "euler")
        integrator_kws.setdefault("dt", 0.01)

        integrator_model = Cuppini2014Integrator(tau=tau, s=s, theta=theta)
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        temporal_filter_model = Cuppini2014TemporalFilter(tau=tau)
        self._temporal_filter = bp.odeint(
            f=temporal_filter_model, **integrator_kws
        )

        self._mode0 = mode0
        self._mode1 = mode1

    # PROPERTY ================================================================

    @property
    def neurons(self):
        return self._neurons

    @property
    def tau(self):
        return self._integrator.f.tau

    @property
    def s(self):
        return self._integrator.f.s

    @property
    def theta(self):
        return self._integrator.f.theta

    @property
    def random(self):
        return self._random

    @property
    def mode0(self):
        return self._mode0

    @property
    def mode1(self):
        return self._mode1

    # Model architecture methods

    def distance(self, position_j, position_k):
        if np.abs(position_j - position_k) <= self.neurons / 2:
            return np.abs(position_j - position_k)
        return self.neurons - np.abs(position_j - position_k)

    def lateral_synapses(
        self,
        excitation_loc,
        inhibition_loc,
        excitation_scale,
        inhibition_scale,
    ):

        the_lateral_synapses = np.zeros((self.neurons, self.neurons))

        for neuron_i in range(self.neurons):
            for neuron_j in range(self.neurons):
                if neuron_i == neuron_j:
                    the_lateral_synapses[neuron_i, neuron_j] = 0
                    continue

                distance = self.distance(neuron_i, neuron_j)
                e_gauss = excitation_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(excitation_scale))
                )
                i_gauss = inhibition_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(inhibition_scale))
                )

                the_lateral_synapses[neuron_i, neuron_j] = e_gauss - i_gauss
        return the_lateral_synapses

    def stimuli_input(self, intensity, *, scale, loc):
        the_stimuli = np.zeros(self.neurons)

        for neuron_j in range(self.neurons):
            distance = self.distance(neuron_j, loc)
            the_stimuli[neuron_j] = intensity * np.exp(
                -(np.square(distance)) / (2 * np.square(scale))
            )

        return the_stimuli

    def create_unimodal_stimuli_matrix(
        self,
        stimuli,
        stimuli_duration,
        onset,
        simulation_length,
        stimuli_n=1,
        soa=None,
    ):
        # TODO expand for more than 2 stimuli and for different stimuli

        # Input before onset
        no_stim = np.zeros(self.neurons)
        pre_stim = np.tile(no_stim, (onset, 1))

        # Input during stimulus delivery
        stim = np.tile(stimuli, (stimuli_duration, 1))

        # If two stimuli are delivered
        if stimuli_n == 2:
            # Input during onset asyncrhony
            soa_stim = np.tile(no_stim, (soa, 1))

            # Input after stimulation
            post_stim_time = (
                simulation_length - onset - stimuli_duration * 2 - soa
            )
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack(
                (pre_stim, stim, soa_stim, stim, post_stim)
            )
            stimuli_matrix = np.repeat(
                complete_stim, 1 / self._integrator.dt, axis=0
            )

        else:
            # Input after stimulation
            post_stim_time = simulation_length - onset - stimuli_duration
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack((pre_stim, stim, post_stim))
            stimuli_matrix = np.repeat(
                complete_stim, 1 / self._integrator.dt, axis=0
            )

        return stimuli_matrix

    def synapses(self, weight):

        the_synapses = np.ones((self.neurons, self.neurons)) * weight

        return the_synapses

    # Model run
    def run(
        self,
        simulation_length,
        soa,
        *,
        onset=16,
        auditory_duration=15,
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
    ):

        if auditory_position == None:
            auditory_position = int(self.neurons / 2)
        if visual_position == None:
            visual_position = int(self.neurons / 2)
        if auditory_gain == None:
            auditory_gain = np.exp(1)
        if visual_gain == None:
            visual_gain = np.exp(1)

        hist_times = np.arange(0, simulation_length, self._integrator.dt)
        sim_cross_modal_latency = int(
            cross_modal_latency / self._integrator.dt
        )

        # Build synapses
        auditory_latsynapses = self.lateral_synapses(
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=24,
        )
        visual_latsynapses = self.lateral_synapses(
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=24,
        )

        auditory_to_visual_synapses = self.synapses(weight=0.35)
        visual_to_auditory_synapses = self.synapses(weight=0.35)

        # Generate Stimuli
        point_auditory_stimuli = self.stimuli_input(
            intensity=auditory_intensity, scale=32, loc=auditory_position
        )
        point_visual_stimuli = self.stimuli_input(
            intensity=visual_intensity, scale=4, loc=visual_position
        )

        auditory_stimuli = self.create_unimodal_stimuli_matrix(
            stimuli=point_auditory_stimuli,
            stimuli_duration=auditory_duration,
            onset=onset,
            simulation_length=simulation_length,
            stimuli_n=2,
            soa=soa,
        )

        visual_stimuli = self.create_unimodal_stimuli_matrix(
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=onset,
            simulation_length=simulation_length,
        )

        # Data holders

        auditory_y, visual_y = (
            np.zeros(self.neurons),
            np.zeros(self.neurons),
        )

        auditory_outside_input, visual_outside_input = (
            np.zeros(self.neurons),
            np.zeros(self.neurons),
        )

        auditory_res, visual_res, multi_res = (
            np.zeros(
                (int(simulation_length / self._integrator.dt), self.neurons)
            ),
            np.zeros(
                (int(simulation_length / self._integrator.dt), self.neurons)
            ),
            np.zeros(
                (int(simulation_length / self._integrator.dt), self.neurons)
            ),
        )

        for i in range(hist_times.size):

            time = hist_times[i]

            # Compute cross-modal input
            auditory_cm_input = np.sum(
                visual_to_auditory_synapses
                * visual_res[i - sim_cross_modal_latency, :],
                axis=1,
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses
                * auditory_res[i - sim_cross_modal_latency, :],
                axis=1,
            )

            # Compute external input
            auditory_input = auditory_stimuli[i] + auditory_cm_input
            visual_input = visual_stimuli[i] + visual_cm_input

            if noise:
                auditory_noise = -(auditory_intensity * 0.4) + (
                    2 * auditory_intensity * 0.4
                ) * self.random.rand(self.neurons)
                visual_noise = -(visual_intensity * 0.4) + (
                    2 * visual_intensity * 0.4
                ) * self.random.rand(self.neurons)
                auditory_input += auditory_noise
                visual_input += visual_noise

            (
                auditory_outside_input,
                visual_outside_input,
            ) = self._temporal_filter(
                a_outside_input=auditory_outside_input,
                v_outside_input=visual_outside_input,
                t=time,
                a_external_input=auditory_stimuli[i],
                v_external_input=visual_stimuli[i],
                a_cross_modal_input=auditory_cm_input,
                v_cross_modal_input=visual_cm_input,
                a_gain=auditory_gain,
                v_gain=visual_gain,
            )

            # Compute lateral inpunt
            la = np.sum(auditory_latsynapses * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses * visual_y, axis=1)

            # Compute unisensory total input
            auditory_u = la + auditory_outside_input
            visual_u = lv + visual_outside_input

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

        return {
            "auditory": auditory_res,
            "visual": visual_res,
            "multi": multi_res,
        }
