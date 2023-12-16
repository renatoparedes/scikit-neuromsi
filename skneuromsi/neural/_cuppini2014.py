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

import copy

from findpeaks import findpeaks

import numpy as np

from ..core import SKNMSIMethodABC


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
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        temporal_filter_model = Cuppini2014TemporalFilter(tau=tau)
        self._temporal_filter = bp.odeint(
            f=temporal_filter_model, **integrator_kws
        )

        self._mode0 = mode0
        self._mode1 = mode1

        self.set_random(np.random.default_rng(seed=seed))

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
    def time_range(self):
        return self._time_range

    @property
    def time_res(self):
        return self._time_res

    @property
    def position_range(self):
        return self._position_range

    @property
    def position_res(self):
        return self._position_res

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

    def lateral_synapse(self, distance, loc, scale):
        return loc * np.exp(-(np.square(distance)) / (2 * np.square(scale)))

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
                distance = self.distance(neuron_i, neuron_j)
                if distance == 0:
                    the_lateral_synapses[neuron_i, neuron_j] = 0
                    continue
                e_gauss = self.lateral_synapse(
                    distance, excitation_loc, excitation_scale
                )
                i_gauss = self.lateral_synapse(
                    distance, inhibition_loc, inhibition_scale
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

        no_stim = np.zeros(self.neurons)

        if stimuli_n == 0:
            stim = np.tile(no_stim, (simulation_length, 1))
            stimuli_matrix = np.repeat(stim, 1 / self._time_res, axis=0)
            return stimuli_matrix

        # Input before onset
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
                complete_stim, 1 / self._time_res, axis=0
            )

        else:
            # Input after stimulation
            post_stim_time = simulation_length - onset - stimuli_duration
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack((pre_stim, stim, post_stim))
            stimuli_matrix = np.repeat(
                complete_stim, 1 / self._time_res, axis=0
            )

        return stimuli_matrix

    # Model run
    def set_random(self, rng):
        self._random = rng

    def compute_latency(self, time, latency):
        if time - latency >= 0:
            return time - latency
        return 0

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

        cross_modal_synapses_weight = 0.35

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
            simulation_length=self._time_range[1],
            stimuli_n=auditory_stim_n,
            soa=soa,
        )

        visual_stimuli = self.create_unimodal_stimuli_matrix(
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=onset,
            simulation_length=self._time_range[1],
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
            computed_cross_latency = self.compute_latency(
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

            # Compute external input

            # auditory_input = auditory_stimuli[i] + auditory_cm_input
            # visual_input = visual_stimuli[i] + visual_cm_input

            # if noise:
            #    auditory_noise = -(auditory_intensity * 0.4) + (
            #        2 * auditory_intensity * 0.4
            #    ) * self.random.rand(self.neurons)
            #    visual_noise = -(visual_intensity * 0.4) + (
            #        2 * visual_intensity * 0.4
            #    ) * self.random.rand(self.neurons)
            #    auditory_input += auditory_noise
            #    visual_input += visual_noise

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
