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

from ..utils.neural_tools import (
    calculate_lateral_synapses,
    calculate_inter_areal_synapses,
    calculate_stimuli_input,
    create_unimodal_stimuli_matrix,
)

from ..utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


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

        self._integrator = bp.odeint(
            f=self._integrator_function, **self._integrator_kws
        )

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
        auditory_duration=None,
        auditory_onset=0,
        auditory_stim_n=1,
        visual_duration=None,
        visual_onset=0,
        visual_stim_n=1,
        noise=False,
        causes_kind="count",
        causes_dim="spatial",
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

        auditory_duration = (
            self._time_range[1]
            if auditory_duration is None
            else auditory_duration
        )

        visual_duration = (
            self._time_range[1] if visual_duration is None else visual_duration
        )

        hist_times = np.arange(
            self._time_range[0], self._time_range[1], self._time_res
        )

        # Build synapses
        auditory_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=5,
            inhibition_loc=4,
            excitation_scale=3,
            inhibition_scale=120,
            dtype=self.dtype,
        )
        visual_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=5,
            inhibition_loc=4,
            excitation_scale=3,
            inhibition_scale=120,
            dtype=self.dtype,
        )
        multi_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=3,
            inhibition_loc=2.6,
            excitation_scale=2,
            inhibition_scale=10,
            dtype=self.dtype,
        )
        auditory_to_visual_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=1.4, sigma=5, dtype=self.dtype
        )
        visual_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=1.4, sigma=5, dtype=self.dtype
        )
        auditory_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=18, sigma=0.5, dtype=self.dtype
        )
        visual_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons, weight=18, sigma=0.5, dtype=self.dtype
        )

        # Generate Stimuli
        point_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_intensity,
            scale=auditory_sigma,
            loc=auditory_position,
            dtype=self.dtype,
        )
        point_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_intensity,
            scale=visual_sigma,
            loc=visual_position,
            dtype=self.dtype,
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
        )

        # Data holders
        y_z = np.zeros(self.neurons, dtype=self.dtype)
        auditory_y, visual_y, multi_y = (
            copy.deepcopy(y_z),
            copy.deepcopy(y_z),
            copy.deepcopy(y_z),
        )

        res_z = np.zeros(
            (int(self._time_range[1] / self._integrator.dt), self.neurons),
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
            auditory_input = auditory_stimuli[i] + auditory_cm_input
            visual_input = visual_stimuli[i] + visual_cm_input
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
            "causes_kind": causes_kind,
            "causes_dim": causes_dim,
            "stim_position": [auditory_position, visual_position],
            "synapses": visual_to_multi_synapses,
        }

        return response, extra

    def calculate_causes(
        self, multi, causes_kind, causes_dim, stim_position, **kwargs
    ):
        # Calculate the average stimuli position
        position = int(np.mean([stim_position[0], stim_position[1]]))

        # Calculates the causes in the desired dimension using the specified method
        causes = calculate_spatiotemporal_causes_from_peaks(
            mode_spatiotemporal_activity_data=multi,
            causes_kind=causes_kind,
            causes_dim=causes_dim,
            score_threshold=0.15,
            time_point=-1,
            spatial_point=position,
        )

        # Return the calculated causes
        return causes
