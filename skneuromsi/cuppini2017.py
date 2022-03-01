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

from .core import SKNMSIMethodABC


@dataclass
class Cuppini2017Integrator:
    tau: tuple
    s: float
    theta: float
    name: str = "Cuppini2017Integrator"

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


class Cuppini2017(SKNMSIMethodABC):
    """Zaraza.


    References
    ----------
    :cite:p:`cuppini2017biologically`

    """

    _sknms_run_method_config = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_intensity", "template": "${mode0}_intensity"},
        {"target": "visual_intensity", "template": "${mode1}_intensity"},
    ]

    def __init__(
        self,
        *,
        neurons=180,
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

        integrator_model = Cuppini2017Integrator(tau=tau, s=s, theta=theta)
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

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
        simulation_length,
        auditory_position,
        visual_position,
        *,
        auditory_intensity=28,
        visual_intensity=27,
        noise=False,
    ):

        hist_times = np.arange(0, simulation_length, self._integrator.dt)

        # Build synapses
        auditory_latsynapses = self.lateral_synapses(
            excitation_loc=5,
            inhibition_loc=4,
            excitation_scale=3,
            inhibition_scale=120,
        )
        visual_latsynapses = self.lateral_synapses(
            excitation_loc=5,
            inhibition_loc=4,
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
        auditory_to_multi_synapses = self.synapses(weight=18, sigma=0.5)
        visual_to_multi_synapses = self.synapses(weight=18, sigma=0.5)

        # Generate Stimuli
        auditory_stimuli = self.stimuli_input(
            intensity=auditory_intensity, scale=32, loc=auditory_position
        )
        visual_simuli = self.stimuli_input(
            intensity=visual_intensity, scale=4, loc=visual_position
        )

        auditory_y, visual_y, multi_y = (
            np.zeros(self.neurons),
            np.zeros(self.neurons),
            np.zeros(self.neurons),
        )

        for time in hist_times:

            # Compute cross-modal input
            auditory_cm_input = np.sum(
                visual_to_auditory_synapses * visual_y, axis=1
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses * auditory_y, axis=1
            )

            # Compute external input
            auditory_input = auditory_stimuli + auditory_cm_input
            visual_input = visual_simuli + visual_cm_input
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

        return auditory_y, visual_y, multi_y


def unisensory_barycenter_readout(
    neurons, auditory_position, visual_position, auditory_y, visual_y
):

    mid = neurons / 2

    if auditory_position < mid:
        abscissa_x = np.concatenate(
            (
                np.arange(auditory_position + mid),
                np.arange(auditory_position - mid, 0),
            )
        )

    if auditory_position > mid:
        abscissa_x = np.concatenate(
            (
                np.arange(neurons, auditory_position + mid),
                np.arange(auditory_position - mid, neurons),
            )
        )

    if auditory_position == mid:
        abscissa_x = np.arange(neurons)

    if visual_position < mid:
        abscissa_y = np.concatenate(
            (
                np.arange(visual_position + mid),
                np.arange(visual_position - mid, 0),
            )
        )

    if visual_position > mid:
        abscissa_y = np.concatenate(
            (
                np.arange(neurons, visual_position + mid),
                np.arange(visual_position - mid, neurons),
            )
        )

    if visual_position == mid:
        abscissa_y = np.arange(neurons)

    auditory_percept = np.sum(auditory_y * abscissa_x) / np.sum(auditory_y)
    visual_percept = np.sum(visual_y * abscissa_y) / np.sum(visual_y)

    return auditory_percept, visual_percept
