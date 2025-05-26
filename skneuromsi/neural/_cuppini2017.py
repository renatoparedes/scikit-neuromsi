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

import brainpy as bp

import numpy as np

from ..core import SKNMSIMethodABC
from ..utils.neural_tools import (
    calculate_inter_areal_synapses,
    calculate_lateral_synapses,
    calculate_stimuli_input,
    create_unimodal_stimuli_matrix,
)
from ..utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


@dataclass
class Cuppini2017Integrator:
    """
    Integrator for the Cuppini2017 model.

    This class represents the integrator function used to compute
    the dynamics of the neural network according to the Cuppini (2017) model.
    It handles the update rules for the neurons' activities based on
    their net inputs.

    """

    #: Time constants for the auditory, visual, and multisensory neurons.
    tau: tuple

    #: Slope of the sigmoid activation function.
    s: float

    #: Central position of the sigmoid activation function.
    theta: float

    #: Name of the integrator
    name: str = "Cuppini2017Integrator"

    @property
    def __name__(self):
        """Return the name of the Integrator."""
        return self.name

    def sigmoid(self, u):
        """Computes the sigmoid activation function."""
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def __call__(self, y_a, y_v, y_m, t, u_a, u_v, u_m):
        """Computes the activites of neurons."""
        # Auditory
        dy_a = (-y_a + self.sigmoid(u_a)) * (1 / self.tau[0])

        # Visual
        dy_v = (-y_v + self.sigmoid(u_v)) * (1 / self.tau[1])

        # Multisensory
        dy_m = (-y_m + self.sigmoid(u_m)) * (1 / self.tau[2])

        return dy_a, dy_v, dy_m


class Cuppini2017(SKNMSIMethodABC):
    r"""Network model for causal inference of Cuppini et al. (2017).

    This model simulates neural responses in a multisensory system,
    consisting of auditory, visual, and multisensory layers.
    The model consists of three layers: two encode auditory and visual stimuli
    separately, and connect to a multisensory layer where causal inference
    is computed. By default, each of these layers consists of 180 neurons
    arranged topologically to encode a 180° space. In this way, each neuron
    encodes 1° of space and neurons close to each other encode
    close spatial positions.


    References
    ----------
    :cite:p:`cuppini2017biologically`

    Notes
    -----
    Each neuron is indicated with a superscript :math:`c` for a specific
    cortical area (:math:`a` for auditory, :math:`v` for visual,
    :math:`m` for multisensory). Each neuron also has a subscript :math:`j`
    referring to its spatial position within a given area.

    The neurons use a sigmoid activation function and first-order dynamics:

    .. math::
      \tau^{c} \frac{dy^{c}_{j}(t)}{dt} = - y^{c}_{j}(t) +
      F(u^{c}_{j}(t)), \ c = a, v, m

    where:

    - :math:`u(t)` and :math:`y(t)` are the net input and output of a neuron at
      time :math:`(t)`.
    - :math:`\tau^{c}` is the time constant of neurons in area :math:`c`.
    - :math:`F(u)` is the sigmoid function:

    .. math::
        F(u_{j}^{c}) = \frac{1}{1 + \exp^{-s (u_{j}^{c} - \theta)}}

    Here, :math:`s` and :math:`\theta` denote the slope and the
    central position of the sigmoid function, respectively.

    Neurons in all regions differ only in their time constants, with faster
    sensory processing for auditory stimuli compared to visual stimuli.

    Neurons are connected in a "Mexican hat" pattern within each layer,
    defined by:

    .. math::
        L_{jk}^{s} = \left\{
        \begin{matrix}
        L_{ex}^{c} \cdot \exp\left(- \frac{(D_{jk})^{2}}
        {2 \cdot (\sigma_{ex}^{c})^{2}}\right) - L_{in}^{c} \cdot
        \exp\left(- \frac{(D_{jk})^{2}}{2 \cdot
        (\sigma_{in}^{c})^{2}}\right), & D_{jk} \neq 0 \\
        0, \ D_{jk} = 0
        \end{matrix}
        \right.

    where:

    - :math:`L_{jk}^{c}` is the synaptic weight from the pre-synaptic neuron
      at position :math:`k` to the post-synaptic neuron at position :math:`j`.
    - :math:`D_{jk}` is the distance between pre-synaptic and
      post-synaptic neurons:

    .. math::
        D_{jk} = \left\{
        \begin{matrix}
        | j-k |, & | j-k | \leqslant N/2 \\
        N - | j-k |, & | j-k | > N/2
        \end{matrix}
        \right.

    Neurons in the unisensory layers are reciprocally connected with
    neurons in the opposite layer (e.g., auditory to visual).
    These connections are defined by:

    .. math::
        W^{cd}_{jk} = W^{cd}_{0} \cdot
        \exp\left(- \frac{(D_{jk})^{2}}{2 (\sigma^{cd})^{2}}\right),
        \ cd = av\text{ or } va

    where:

    - :math:`W_{0}^{cd}` is the maximum synaptic efficacy.
    - :math:`D_{jk}` is the distance between neurons
      in different sensory regions.
    - :math:`\sigma^{cd}` is the width of the cross-modal synapses.

    Neurons in unisensory layers also have excitatory connections
    to the multisensory layer:

    .. math::
        W^{mc}_{jk} = W^{mc}_{0} \cdot
        \exp\left(- \frac{(D_{jk})^{2}}{2 (\sigma^{mc})^{2}}\right),
        \ c = a,v

    where:

    - :math:`W^{mc}_{0}` is the highest value of synaptic efficacy.
    - :math:`D_{jk}` is the distance between neurons in multisensory and
      unisensory areas.
    - :math:`\sigma^{mc}` is the width of the feedforward synapses.

    The visual and auditory stimuli are modeled as Gaussian functions:

    .. math::
        e^{c}_{j}(t) = E^{c}_{0} \cdot
        \exp\left(- \frac{(d^{c}_{j})^{2}}{2 (\sigma^{c})^{2}}\right)

    where:

    - :math:`E^{c}_{0}` is the stimulus strength.
    - :math:`d^{c}_{j}` is the distance between the neuron and the stimulus.
    - :math:`\sigma^{c}` is the degree of uncertainty in detection.

    The net input to a neuron combines within-region and extra-area components:

    .. math::
        u^{c}_{j}(t) = l^{c}_{j}(t) + o^{c}_{j}(t)

    where :math:`l^{c}_{j}(t)` is the within-region input:

    .. math::
        l^{c}_{j}(t) = \sum_{k} L^{c}_{jk} \cdot y^{c}_{jk}(t)

    Here :math:`o^{c}_{j}(t)` is the extra-area input,
    including external stimuli, cross-modal input, and noise.

    """

    _model_name = "Cuppini2017"
    _model_type = "Neural"
    _run_input = [
        {"target": "auditory_position", "template": "${mode0}_position"},
        {"target": "visual_position", "template": "${mode1}_position"},
        {"target": "auditory_intensity", "template": "${mode0}_intensity"},
        {"target": "visual_intensity", "template": "${mode1}_intensity"},
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
        """
        Initializes the Cuppini2017 model.

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
        _integrator_function : Cuppini2017IntegratorFunction
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

        integrator_model = Cuppini2017Integrator(tau=tau, s=s, theta=theta)
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        self.set_random(np.random.default_rng(seed=seed))

        self._mode0 = mode0
        self._mode1 = mode1

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
        return self._integrator_function.tau

    @property
    def s(self):
        """
        Slope of the sigmoid activation function.

        Returns
        -------
        float
            The slope parameter of the sigmoid function used in the model.
        """
        return self._integrator_function.s

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
        return self._integrator_function.theta

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
        auditory_soa=None,
        visual_soa=None,
        noise=False,
        noise_level=0.40,
        feedforward_weight=18,
        cross_modal_weight=1.4,
        causes_kind="count",
        causes_dim="space",
        causes_peak_threshold=0.15,
        causes_peak_distance=None,
    ):
        """
        Run the simulation of the Cuppini2017 model.

        It computes the activity of auditory, visual, and multisensory neurons
        based on the provided inputs and parameters. The simulation includes
        the generation of stimuli, calculation of lateral and cross-modal
        synaptic inputs, and the integration of the model's dynamics.

        Parameters
        ----------
        auditory_position : int, optional
            The spatial position of the auditory stimulus.
            Defaults to the center of the position range if not provided.
        visual_position : int, optional
            The spatial position of the visual stimulus.
            Defaults to the center of the position range if not provided.
        auditory_sigma : float, optional
            The standard deviation of the Gaussian function
            used to represent the auditory stimulus. Default is 32.
        visual_sigma : float, optional
            The standard deviation of the Gaussian function
            used to represent the visual stimulus. Default is 4.
        auditory_intensity : float, optional
            The intensity of the auditory stimulus. Default is 28.
        visual_intensity : float, optional
            The intensity of the visual stimulus. Default is 27.
        auditory_duration : float, optional
            The duration of the auditory stimulus.
            Defaults to the full time range if not provided.
        auditory_onset : float, optional
            The onset time of the auditory stimulus. Default is 0.
        auditory_stim_n : int, optional
            The number of auditory stimuli to generate. Default is 1.
        visual_duration : float, optional
            The duration of the visual stimulus.
            Defaults to the full time range if not provided.
        visual_onset : float, optional
            The onset time of the visual stimulus. Default is 0.
        visual_stim_n : int, optional
            The number of visual stimuli to generate. Default is 1.
        auditory_soa : float, optional
            Stimulus-onset asynchrony for auditory stimuli (default is None).
        visual_soa : float, optional
            Stimulus-onset asynchrony for visual stimuli (default is None).
        noise : bool, optional
            Whether to include noise in the simulation. Default is False.
        noise_level : float, optional
            Level of noise to add (default is 0.40).
        feedforward_weight : float, optional
            The weight of the feedforward synapses from the unisensory areas
            to the multisensory area. Default is 18.
        cross_modal_weight : float, optional
            The weight of the cross-modal synapses between unisensory areas.
            Default is 1.4.
        causes_kind : str, optional
            The method used to calculate causes. Default is "count".
        causes_dim : str, optional
            The dimension in which to calculate causes. Default is "space".
        causes_peak_threshold : float, optional
            Peak threshold for causes calculation (default is 0.15).

        Returns
        -------
        tuple
            A tuple containing two elements:
            - `response` (dict): A dictionary with keys "auditory", "visual",
            and "multi" containing the results of the simulation for auditory,
            visual, and multisensory neurons respectively.
            - `extra` (dict): A dictionary with additional information,
            including causes parameters, and stimulus positions.
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

        auditory_duration = (
            self._time_range[1]
            if auditory_duration is None
            else auditory_duration
        )

        visual_duration = (
            self._time_range[1] if visual_duration is None else visual_duration
        )

        hist_times = np.arange(
            self._time_range[0], self._time_range[1], self._integrator.dt
        )

        n_time_steps = hist_times.size

        # Build synapses
        auditory_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=5,
            inhibition_loc=4,
            excitation_scale=3,
            inhibition_scale=120,
        )
        visual_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=5,
            inhibition_loc=4,
            excitation_scale=3,
            inhibition_scale=120,
        )
        multi_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons,
            excitation_loc=3,
            inhibition_loc=2.6,
            excitation_scale=2,
            inhibition_scale=10,
        )
        auditory_to_visual_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=cross_modal_weight,
            sigma=5,
        )
        visual_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=cross_modal_weight,
            sigma=5,
        )
        auditory_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=feedforward_weight,
            sigma=0.5,
        )
        visual_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=feedforward_weight,
            sigma=0.5,
        )

        # Generate Stimuli
        point_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_intensity,
            scale=auditory_sigma,
            loc=auditory_position,
        )
        point_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_intensity,
            scale=visual_sigma,
            loc=visual_position,
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

        auditory_input, visual_input, multi_input = (
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
            copy.deepcopy(z_1d),
        )

        z_2d = np.zeros((n_time_steps, self.neurons))
        auditory_res, visual_res, multi_res = (
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
            copy.deepcopy(z_2d),
        )

        del z_1d, z_2d

        for i in range(n_time_steps):
            time = int(hist_times[i] / self._integrator.dt)

            # Compute cross-modal input
            auditory_cm_input = np.sum(
                visual_to_auditory_synapses * visual_y, axis=1
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses * auditory_y, axis=1
            )

            # Compute feedforward input
            multi_input = np.sum(
                auditory_to_multi_synapses * auditory_y, axis=1
            ) + np.sum(visual_to_multi_synapses * visual_y, axis=1)

            # Compute external input
            auditory_input = auditory_stimuli[i] + auditory_cm_input
            visual_input = visual_stimuli[i] + visual_cm_input

            # Input noise
            if noise:
                auditory_noise = -(auditory_intensity * noise_level) + (
                    2 * auditory_intensity * noise_level
                ) * self.random.random(self.neurons)
                visual_noise = -(visual_intensity * noise_level) + (
                    2 * visual_intensity * noise_level
                ) * self.random.random(self.neurons)
                auditory_input += auditory_noise
                visual_input += visual_noise

            # Compute lateral input
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
        Calculate the causes based on spatiotemporal activity peaks.

        This method computes the causes (i.e., the underlying factors
        or sources) of multisensory activity based on the peaks in the
        multisensory data. The calculation considers the specified method and
        dimension for cause determination.

        Parameters
        ----------
        multi : array_like
            A 2D array of shape (time_points, neurons) representing
            the multisensory activity data from the simulation.
            The data is used to determine the spatiotemporal causes.
        causes_kind : str
            The method used to determine the causes. This could refer to
            the type of analysis or metric for identifying causes.
            For example, it might specify whether to count occurrences or use
            some other measure.
        causes_dim : str
            The dimension in which to calculate causes. This specifies whether
            the causes should be determined based on spatial or temporal peaks.
        causes_peak_threshold : float
            Peak threshold for causes calculation.
        stim_position : list of int
            A list containing the positions of the auditory and
            visual stimuli used in the simulation. These positions help in
            calculating the causes by providing context on where the stimuli
            were located.
        **kwargs
            Additional keyword arguments to be passed to
            the cause calculation function.

        Returns
        -------
        causes : dict
            A dictionary containing the calculated causes. The structure of
            the dictionary will depend on the `causes_kind` and
            `causes_dim` parameters. It generally includes information about
            the detected causes based on the spatiotemporal activity data.
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
