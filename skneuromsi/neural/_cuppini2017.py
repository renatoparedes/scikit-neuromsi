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

from ..utils.readout_tools import calculate_single_peak_probability


@dataclass
class Cuppini2017IntegratorFunction:
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

    _model_name = "Cuppini2017"
    _model_type = "Neural"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_intensity", "template": "${mode0}_intensity"},
        {"target": "visual_intensity", "template": "${mode1}_intensity"},
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
        tau=(3, 15, 1),
        s=0.3,
        theta=20,
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

        self._integrator_function = Cuppini2017IntegratorFunction(
            tau=tau, s=s, theta=theta
        )
        self._integrator_kws = integrator_kws

        self._mode0 = mode0
        self._mode1 = mode1

        self.set_random(np.random.default_rng(seed=seed))

        self._dtype = np.float32

    # PROPERTY ================================================================

    @property
    def neurons(self):
        return self._neurons

    @property
    def tau(self):
        return self._integrator_function.tau

    @property
    def s(self):
        return self._integrator_function.s

    @property
    def theta(self):
        return self._integrator_function.theta

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

    @property
    def dtype(self):
        return self._dtype

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
        the_lateral_synapses = np.zeros(
            (self.neurons, self.neurons), dtype=self.dtype
        )

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
        the_stimuli = np.zeros(self.neurons, dtype=self.dtype)

        for neuron_j in range(self.neurons):
            distance = self.distance(neuron_j, loc)
            the_stimuli[neuron_j] = intensity * np.exp(
                -(np.square(distance)) / (2 * np.square(scale))
            )

        return the_stimuli

    def synapses(self, weight, sigma):
        the_synapses = np.zeros((self.neurons, self.neurons), dtype=self.dtype)

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

    def run(
        self,
        *,
        auditory_position=None,
        visual_position=None,
        auditory_sigma=32,
        visual_sigma=4,
        auditory_intensity=28,
        visual_intensity=27,
        noise=False,
        causes_kind="count",
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

        hist_times = np.arange(
            self._time_range[0], self._time_range[1], self._time_res
        )

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
            intensity=auditory_intensity,
            scale=auditory_sigma,
            loc=auditory_position,
        )
        visual_stimuli = self.stimuli_input(
            intensity=visual_intensity, scale=visual_sigma, loc=visual_position
        )

        # create the integrator
        integrator = bp.odeint(
            f=self._integrator_function, **self._integrator_kws
        )

        # Data holders
        y_z = np.zeros(self.neurons, dtype=self.dtype)
        auditory_y, visual_y, multi_y = (
            copy.deepcopy(y_z),
            copy.deepcopy(y_z),
            copy.deepcopy(y_z),
        )

        res_z = np.zeros(
            (int(self._time_range[1] / integrator.dt), self.neurons),
            dtype=self.dtype,
        )
        auditory_res, visual_res, multi_res = (
            copy.deepcopy(res_z),
            copy.deepcopy(res_z),
            copy.deepcopy(res_z),
        )

        auditory_noise = -(auditory_intensity * 0.4) + (
            2 * auditory_intensity * 0.4
        ) * self.random.random(self.neurons)
        visual_noise = -(visual_intensity * 0.4) + (
            2 * visual_intensity * 0.4
        ) * self.random.random(self.neurons)

        for i in range(hist_times.size):
            time = hist_times[i]

            # Compute cross-modal input
            auditory_cm_input = np.sum(
                visual_to_auditory_synapses * visual_y, axis=1
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses * auditory_y, axis=1
            )

            # Compute external input
            auditory_input = auditory_stimuli + auditory_cm_input
            visual_input = visual_stimuli + visual_cm_input
            multi_input = np.sum(
                auditory_to_multi_synapses * auditory_y, axis=1
            ) + np.sum(visual_to_multi_synapses * visual_y, axis=1)

            if noise:
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
            auditory_y, visual_y, multi_y = integrator(
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

        extra = {"causes_kind": causes_kind}

        return response, extra

    def calculate_causes(self, multi, causes_kind, **kwargs):
        # Define the topology method to identify the peaks
        fp = findpeaks(method="topology", verbose=0)

        # Get the last data from the multi matrix
        X = multi[-1, :]

        # Find the peaks in the data and get a DataFrame with the results
        fp_results = fp.fit(X)
        multi_peaks_df = fp_results["df"].query(
            "peak==True & valley==False & score>0.15"
        )

        # Determine the type of cause to calculate
        if causes_kind == "count":
            # If counting the number of causes, assign the number of peaks found
            peaks = multi_peaks_df.score.size
        elif causes_kind == "prob":
            # If calculating the probability of a unique cause,
            # calculate the probability of detecting a single peak
            peaks = calculate_single_peak_probability(
                multi_peaks_df["y"].values
            )
        else:
            # If no valid cause type is specified, assign None
            peaks = None

        # Return the calculated causes
        return peaks
