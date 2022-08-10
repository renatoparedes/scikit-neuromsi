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
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau[0])

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau[1])

        # Multisensory
        dy_m = (-y_m + self.sigmoid(u_m)) * (1 / self.tau[2])

        return dy_a, dy_v, dy_m


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
    ]
    _run_output = [
        {"target": "auditory", "template": "${mode0}"},
        {"target": "visual", "template": "${mode1}"},
    ]

    def __init__(
        self,
        *,
        neurons=90,
        tau=(3, 15, 1),
        s=0.3,
        theta=20,
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

        integrator_model = Paredes2022Integrator(tau=tau, s=s, theta=theta)
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

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
    def run(
        self,
        *,
        simulation_length=100,
        soa=50,
        onset=16,
        auditory_duration=7,
        visual_duration=12,
        auditory_position=None,
        visual_position=None,
        auditory_intensity=28,
        visual_intensity=27,
        noise=False,
        lateral_excitation=5,
        lateral_inhibition=4,
    ):

        if auditory_position == None:
            auditory_position = int(self.neurons / 2)
        if visual_position == None:
            visual_position = int(self.neurons / 2)

        hist_times = np.arange(0, simulation_length, self._integrator.dt)

        # Build synapses
        auditory_latsynapses = self.lateral_synapses(
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=120,
        )
        visual_latsynapses = self.lateral_synapses(
            excitation_loc=lateral_excitation,
            inhibition_loc=lateral_inhibition,
            excitation_scale=3,
            inhibition_scale=120,
        )
        multi_latsynapses = self.lateral_synapses(
            excitation_loc=3,
            inhibition_loc=2.6,
            excitation_scale=2,
            inhibition_scale=10,
        )
        auditory_to_visual_synapses = self.synapses(weight=1.4, sigma=5)
        visual_to_auditory_synapses = self.synapses(weight=1.4, sigma=5)
        auditory_to_multi_synapses = self.synapses(weight=22.5, sigma=0.5)
        visual_to_multi_synapses = self.synapses(weight=22.5, sigma=0.5)
        multi_to_auditory_synapses = self.synapses(weight=13.5, sigma=0.5)
        multi_to_visual_synapses = self.synapses(weight=13.5, sigma=0.5)

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

        auditory_y, visual_y, multi_y = (
            np.zeros(self.neurons),
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
                visual_to_auditory_synapses * visual_y, axis=1
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses * auditory_y, axis=1
            )

            # Compute feedback input
            auditory_feedback_input = np.sum(
                multi_to_auditory_synapses * multi_y, axis=1
            )
            visual_feedback_input = np.sum(
                multi_to_visual_synapses * multi_y, axis=1
            )

            # Compute external input
            auditory_input = (
                auditory_stimuli[i]
                + auditory_cm_input
                + auditory_feedback_input
            )
            visual_input = (
                visual_stimuli[i] + visual_cm_input + visual_feedback_input
            )
            multi_input = np.sum(
                auditory_to_multi_synapses * auditory_y, axis=1
            ) + np.sum(visual_to_multi_synapses * visual_y, axis=1)

            if noise:
                auditory_noise = -(auditory_intensity * 0.4) + (
                    2 * auditory_intensity * 0.4
                ) * self.random.rand(self.neurons)
                visual_noise = -(visual_intensity * 0.4) + (
                    2 * visual_intensity * 0.4
                ) * self.random.rand(self.neurons)
                auditory_input += auditory_noise
                visual_input += visual_noise

            # Compute lateral inpunt
            la = np.sum(auditory_latsynapses * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses * visual_y, axis=1)
            lm = np.sum(multi_latsynapses * multi_y, axis=1)

            # Compute unisensory total input
            auditory_u = la + auditory_input
            visual_u = lv + visual_input

            # Compute multisensory total input
            u_m = lm + multi_input

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
        return response, {}
