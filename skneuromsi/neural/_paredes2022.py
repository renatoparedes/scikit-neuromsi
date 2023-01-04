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

import numpy as np

from scipy.signal import find_peaks

from ..core import SKNMSIMethodABC


@dataclass
class Paredes2022Integrator:
    tau: tuple
    s: float
    theta: float
    name: str = "Paredes2022Integrator"

    @property
    def __name__(self):
        return self.name

    def sigmoid(self, u):
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def __call__(self, y_a, y_v, y_m, t, u_a, u_v, u_m):

        # Auditory
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau)

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau)

        # Multisensory
        dy_m = (-y_m + self.sigmoid(u_m)) * (1 / self.tau)

        return dy_a, dy_v, dy_m


@dataclass
class Paredes2022TemporalFilter:
    tau: tuple
    name: str = "Paredes2022TemporalFilter"

    @property
    def __name__(self):
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
    ):

        # Auditory
        da_outside_input = auditoryfilter_input

        dauditory_filter_input = (
            (a_gain / self.tau[0])
            * (a_external_input + a_cross_modal_input + a_feedback_input)
            - ((2 * auditoryfilter_input) / self.tau[0])
            - a_outside_input / np.square(self.tau[0])
        )

        # Visual
        dv_outside_input = visualfilter_input

        dvisual_filter_input = (
            (v_gain / self.tau[1])
            * (v_external_input + v_cross_modal_input + v_feedback_input)
            - ((2 * visualfilter_input) / self.tau[1])
            - v_outside_input / np.square(self.tau[1])
        )

        # Multisensory
        dm_outside_input = multisensoryfilter_input

        dmultisensory_filter_input = (
            (m_gain / self.tau[2]) * (m_external_input)
            - ((2 * multisensoryfilter_input) / self.tau[2])
            - m_outside_input / np.square(self.tau[2])
        )

        return (
            da_outside_input,
            dv_outside_input,
            dm_outside_input,
            dauditory_filter_input,
            dvisual_filter_input,
            dmultisensory_filter_input,
        )


class Paredes2022(SKNMSIMethodABC):
    """Zaraza.


    References
    ----------
    :cite:p:`cuppini2017biologically`

    """

    _model_name = "Paredes2022"
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
        if len(tau) != 3:
            raise ValueError()

        self._neurons = neurons
        self._position_range = position_range
        self._position_res = float(position_res)
        self._time_range = time_range
        self._time_res = float(time_res)

        integrator_kws.setdefault("method", "euler")
        integrator_kws.setdefault("dt", self._time_res)

        integrator_model = Paredes2022Integrator(
            tau=tau_neurons, s=s, theta=theta
        )
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        temporal_filter_model = Paredes2022TemporalFilter(tau=tau)
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
        no_stim = np.zeros(self.neurons)

        if stimuli_n == 0:
            stim = np.tile(no_stim, (self._time_range[1], 1))
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

    def synapses(self, weight, sigma):

        the_synapses = np.zeros((self.neurons, self.neurons))

        for j in range(self.neurons):
            for k in range(self.neurons):
                d = self.distance(j, k)
                the_synapses[j, k] = weight * np.exp(
                    -(np.square(d)) / (2 * np.square(sigma))
                )
        return the_synapses

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
        soa=50,
        auditory_onset=16,
        visual_onset=16,
        auditory_duration=7,
        visual_duration=12,
        auditory_position=None,
        visual_position=None,
        auditory_intensity=1.5,
        visual_intensity=1.1,
        noise=False,
        lateral_excitation=2,
        lateral_inhibition=1.8,
        cross_modal_latency=16,
        feed_latency=95,
        auditory_gain=None,
        visual_gain=None,
        multisensory_gain=None,
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
        multi_latsynapses = self.lateral_synapses(
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=24,
        )
        auditory_to_visual_synapses = self.synapses(weight=0.075, sigma=5)
        visual_to_auditory_synapses = self.synapses(weight=0.075, sigma=5)
        auditory_to_multi_synapses = self.synapses(weight=1.4, sigma=0.5)
        visual_to_multi_synapses = self.synapses(weight=1.4, sigma=0.5)
        multi_to_auditory_synapses = self.synapses(weight=0.10, sigma=0.5)
        multi_to_visual_synapses = self.synapses(weight=0.10, sigma=0.5)

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
            onset=auditory_onset,
            simulation_length=self._time_range[1],
            stimuli_n=auditory_stim_n,
            soa=soa,
        )

        visual_stimuli = self.create_unimodal_stimuli_matrix(
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=visual_onset,
            simulation_length=self._time_range[1],
            stimuli_n=visual_stim_n,
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
            auditory_outside_inputs,
            visual_outside_inputs,
            multisensory_outside_inputs,
        ) = (copy.deepcopy(z_2d), copy.deepcopy(z_2d), copy.deepcopy(z_2d))
        (
            auditoryfilter_inputs,
            visualfilter_inputs,
            multisensoryfilter_inputs,
        ) = (copy.deepcopy(z_2d), copy.deepcopy(z_2d), copy.deepcopy(z_2d))

        (
            auditory_lateral_inputs,
            visual_lateral_inputs,
            multisensory_lateral_inputs,
        ) = (copy.deepcopy(z_2d), copy.deepcopy(z_2d), copy.deepcopy(z_2d))
        (
            auditory_total_inputs,
            visual_total_inputs,
            multisensory_total_inputs,
        ) = (copy.deepcopy(z_2d), copy.deepcopy(z_2d), copy.deepcopy(z_2d))

        del z_1d, z_2d

        auditory_noise = -(auditory_intensity * 0.4) + (
            2 * auditory_intensity * 0.4
        ) * self.random.random(self.neurons)
        visual_noise = -(visual_intensity * 0.4) + (
            2 * visual_intensity * 0.4
        ) * self.random.random(self.neurons)

        for i in range(hist_times.size):

            time = int(hist_times[i] / self._integrator.dt)

            # Compute cross-modal input
            computed_cross_latency = self.compute_latency(
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
            computed_feed_latency = self.compute_latency(
                time, sim_feed_latency
            )

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
            )

            (
                auditory_outside_inputs[i, :],
                visual_outside_inputs[i, :],
                multisensory_outside_inputs[i, :],
            ) = (
                auditory_outside_input,
                visual_outside_input,
                multisensory_outside_input,
            )

            (
                auditoryfilter_inputs[i, :],
                visualfilter_inputs[i, :],
                multisensoryfilter_inputs[i, :],
            ) = (
                auditoryfilter_input,
                visualfilter_input,
                multisensoryfilter_input,
            )

            # if noise:
            #    auditory_input += auditory_noise
            #    visual_input += visual_noise

            # Compute lateral inpunt
            la = np.sum(auditory_latsynapses.T * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses.T * visual_y, axis=1)
            lm = np.sum(multi_latsynapses.T * multi_y, axis=1)

            (
                auditory_lateral_inputs[i, :],
                visual_lateral_inputs[i, :],
                multisensory_lateral_inputs[i, :],
            ) = (la, lv, lm)

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
        return response, {
            "auditory_total_input": auditory_total_inputs,
            "visual_total_input": visual_total_inputs,
            "multi_total_input": multisensory_total_inputs,
        }

    def calculate_perceived_positions(self, auditory, visual, multi, **kwargs):
        a = auditory[-1, :].argmax()
        v = visual[-1, :].argmax()
        m = multi[-1, :].argmax()

        return {
            "perceived_auditory_position": a,
            "perceived_visual_position": v,
            "perceived_multi_position": m,
        }

    def calculate_causes(
        self, multi, **kwargs
    ):  # TODO Include causes for space and time

        # if dimension == "space":
        #    peaks_idx, _ = find_peaks(multi[-1, :], prominence=0.30, height=0.40)
        #    peaks = np.size(peaks_idx)
        #    return peaks

        position = int(self._position_range[1] / 2)
        peaks_idx, _ = find_peaks(
            multi[:, position], prominence=0.30, height=0.40
        )
        peaks = np.size(peaks_idx)
        return peaks
