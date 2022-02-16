from dataclasses import dataclass

import brainpy as bp
import numpy as np


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

    def __call__(self, Y_a, Y_v, Y_m, t, U_a, U_v, U_m):

        # Auditory
        dY_a = (-Y_a + self.sigmoid(U_a)) * (1 / self.tau[0])

        # Visual
        dY_v = (-Y_v + self.sigmoid(U_v)) * (1 / self.tau[1])

        # Multisensory
        dY_m = (-Y_m + self.sigmoid(U_m)) * (1 / self.tau[2])

        return dY_a, dY_v, dY_m


class Cuppini2017:
    def __init__(
        self,
        *,
        neurons=180,
        tau=(3, 15, 1),
        s=0.3,
        theta=20,
        seed=None,
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
                    -(distance**2) / (2 * excitation_scale**2)
                )
                i_gauss = inhibition_loc * np.exp(
                    -(distance**2) / (2 * inhibition_scale**2)
                )

                the_lateral_synapses[neuron_i, neuron_j] = e_gauss - i_gauss
        return the_lateral_synapses

    def stimuli_input(self, intensity, *, scale, loc):
        the_stimuli = np.zeros(self.neurons)

        for neuron_j in range(self.neurons):
            distance = self.distance(neuron_j, loc)
            the_stimuli[neuron_j] = intensity * np.exp(
                -(distance**2) / (2 * scale**2)
            )

        return the_stimuli

    def synapses(self, weight, sigma):

        the_synapses = np.zeros((self.neurons, self.neurons))

        for j in range(self.neurons):
            for k in range(self.neurons):
                d = self.distance(j, k)
                the_synapses[j, k] = weight * np.exp(
                    -(d**2) / (2 * sigma**2)
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
        multi_to_auditory_synapses = self.synapses(weight=18, sigma=0.5)
        multi_to_visual_synapses = self.synapses(weight=18, sigma=0.5)

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

        # Compute cross-modal input
        auditory_cm_input = np.sum(
            auditory_to_visual_synapses * auditory_y, axis=1
        )
        visual_cm_input = np.sum(
            visual_to_auditory_synapses * visual_y, axis=1
        )

        for time in hist_times:

            # Compute external input
            auditory_input = auditory_stimuli + auditory_cm_input
            visual_input = visual_simuli + visual_cm_input
            multi_input = np.sum(
                multi_to_auditory_synapses * auditory_y, axis=1
            ) + np.sum(multi_to_visual_synapses * visual_y, axis=1)

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
            U_m = lm + multi_input

            # Compute neurons activity
            auditory_y, visual_y, multi_y = self._integrator(
                Y_a=auditory_y,
                Y_v=visual_y,
                Y_m=multi_y,
                t=time,
                U_a=auditory_u,
                U_v=visual_u,
                U_m=U_m,
            )

        return auditory_y, visual_y, multi_y
