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

from findpeaks import findpeaks

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
        a_noise,
        v_noise,
        include_noise,
        include_temporal_noise,
        a_temporal_noise,
        v_temporal_noise,
        m_temporal_noise,
    ):
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

        self.set_random(np.random.default_rng(seed=seed))

        self._mode0 = mode0
        self._mode1 = mode1

    # PROPERTY ================================================================

    @property
    def neurons(self):
        return self._neurons

    @property
    def tau_neurons(self):
        return self._integrator.f.tau

    @property
    def tau(self):
        return self._temporal_filter.f.tau

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
        if soa is not None:
            soa = int(soa)

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
            soa_stim = np.tile(no_stim, (soa - stimuli_duration, 1))

            # Input after stimulation
            post_stim_time = (
                simulation_length
                - onset
                - stimuli_duration * 2
                - (soa - stimuli_duration)
            )
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack(
                (pre_stim, stim, soa_stim, stim, post_stim)
            )
            stimuli_matrix = np.repeat(
                complete_stim, 1 / self._integrator.dt, axis=0
            )

        elif stimuli_n == 3:
            # Input during onset asyncrhony
            soa_stim = np.tile(no_stim, (soa - stimuli_duration, 1))

            # Input after stimulation
            post_stim_time = (
                simulation_length
                - onset
                - stimuli_duration * 3
                - (soa - stimuli_duration) * 2
            )
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack(
                (pre_stim, stim, soa_stim, stim, soa_stim, stim, post_stim)
            )
            stimuli_matrix = np.repeat(
                complete_stim, 1 / self._integrator.dt, axis=0
            )

        elif stimuli_n == 4:
            # Input during onset asyncrhony
            soa_stim = np.tile(no_stim, (soa - stimuli_duration, 1))

            # Input after stimulation
            post_stim_time = (
                simulation_length
                - onset
                - stimuli_duration * 4
                - (soa - stimuli_duration) * 3
            )
            post_stim = np.tile(no_stim, (post_stim_time, 1))

            # Input concatenation
            complete_stim = np.vstack(
                (
                    pre_stim,
                    stim,
                    soa_stim,
                    stim,
                    soa_stim,
                    stim,
                    soa_stim,
                    stim,
                    post_stim,
                )
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

    def pruning(self, synapses_weight_matrix, pruning_threshold):
        pruned_synpases = np.copy(synapses_weight_matrix)
        pruned_synpases[pruned_synpases < pruning_threshold] = 0

        return pruned_synpases

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
        auditory_intensity=2.4,
        visual_intensity=1.4,
        noise=False,
        noise_level=0.40,
        temporal_noise=False,
        temporal_noise_scale=5,
        lateral_excitation=2,
        lateral_inhibition=1.8,
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
        auditory_to_visual_synapses = self.synapses(
            weight=cross_modal_weight, sigma=5
        )
        visual_to_auditory_synapses = self.synapses(
            weight=cross_modal_weight, sigma=5
        )
        auditory_to_multi_synapses = self.synapses(
            weight=feedforward_weight, sigma=0.5
        )
        visual_to_multi_synapses = self.synapses(
            weight=feedforward_weight, sigma=0.5
        )
        multi_to_auditory_synapses = self.synapses(
            weight=feedback_weight, sigma=0.5
        )
        multi_to_visual_synapses = self.synapses(
            weight=feedback_weight, sigma=0.5
        )

        # Prune synapses
        auditory_to_multi_synapses = self.pruning(
            auditory_to_multi_synapses, feedforward_pruning_threshold
        )
        visual_to_multi_synapses = self.pruning(
            visual_to_multi_synapses, feedforward_pruning_threshold
        )
        auditory_to_visual_synapses = self.pruning(
            auditory_to_visual_synapses, cross_modal_pruning_threshold
        )
        visual_to_auditory_synapses = self.pruning(
            visual_to_auditory_synapses, cross_modal_pruning_threshold
        )

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
                a_noise=auditory_noise,
                v_noise=visual_noise,
                include_noise=noise,
                include_temporal_noise=temporal_noise,
                a_temporal_noise=rand_a_tau,
                v_temporal_noise=rand_v_tau,
                m_temporal_noise=rand_m_tau,
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

    def calculate_causes(
        self, multi, **kwargs
    ):  # TODO Include causes for space and time
        # if dimension == "space":

        fp = findpeaks(method="topology", verbose=0)
        position = int(self._position_range[1] / 2)
        X = multi[:, position]
        results = fp.fit(X)
        peaks = results["df"].query("peak==True & score>0.4").score.size

        return peaks
