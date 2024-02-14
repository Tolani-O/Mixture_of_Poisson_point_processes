import numpy as np


class DataAnalyzer:

    def __init__(self):
        self.time = None
        self.latent_factors = None
        self.intensity = None
        self.binned = None
        self.latent_coupling = None


    def initialize(self, K=100, T=200, intensity_type=('constant', '1peak', '2peaks'),
                   coeff=(1, 1, 1), ratio=(1/3, 1/3, 1/3), intensity_mltply=15,
                   intensity_bias=5, trial_offsets=np.arange(50)):
        self.time = np.arange(0, T, 1) / 100
        self.latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        self.intensity, self.binned, self.latent_coupling = self.generate_spike_trains(coeff, ratio, K, trial_offsets)
        return self

    def generate_latent_factors(self, intensity_type, intensity_mltply, intensity_bias):

        # intensity_type is a string
        if isinstance(intensity_type, str):
            intensity_type = [intensity_type]

        time = self.time
        num_timesteps = len(time)
        latent_factors = np.zeros((len(intensity_type), num_timesteps))
        for i, itype in enumerate(intensity_type):
            rate = np.zeros_like(time)

            if itype == 'constant':
                nothing_to_do = True

            elif itype == 'sine':
                # Generate y values for sine function with period 2
                rate = np.sin(2 * np.pi * time / 2)

            elif itype == '1peak':
                rate[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
                rate[time > 0.625] = 0

            elif itype == '2peaks':
                rate[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
                rate[time > 0.625] = 0
                rate[time >= 1.25] = np.sin(2 * np.pi * (time[time >= 1.25] - 1.25) / 0.75)
                rate[time > 1.625] = 0

            latent_factors[i, :] = intensity_mltply * rate + intensity_bias

        return latent_factors


    def generate_spike_trains(self, coeff, ratio, num_neurons, trial_offsets):

        latent_factors = self.latent_factors
        num_factors, num_timesteps = latent_factors.shape
        if num_factors == 1:
            ratio = [1]
        # make sure intensity_type and ratio have the same length
        assert num_factors == len(ratio)
        # make sure ratio sums to 1
        ratio = np.array(ratio)
        ratio = ratio / np.sum(ratio)
        num_trials = len(trial_offsets)
        intensity = np.zeros((num_neurons, num_trials, num_timesteps))
        latent_coupling = np.zeros((num_neurons, num_factors))
        binned = np.zeros((num_neurons, num_trials, num_timesteps))
        dt = round(self.time[1] - self.time[0], 3)
        last_binned_index = 0
        # loop over the rows of latent_factors
        for i in range(num_factors):
            neuron_count = int(num_neurons * ratio[i])
            intensity[last_binned_index:(last_binned_index + neuron_count), :, :] = np.stack([np.vstack([coeff[i] * latent_factors[i, :]] * num_trials)] * neuron_count, axis=0)
            latent_coupling[last_binned_index:(last_binned_index + neuron_count), i] = 1
            binned[last_binned_index:(last_binned_index + neuron_count), :, :] = (
                np.random.poisson(intensity[last_binned_index:(last_binned_index + neuron_count), :, :] * dt))
            last_binned_index += neuron_count
        for j in trial_offsets:
            intensity[:, j, :] = np.roll(intensity[:, j, :], j, axis=1) #np.random.normal(0, 0.1, (num_neurons, num_timesteps)))
            binned[:, j, :] = np.roll(binned[:, j, :], j, axis=1)

        # pick only the first last_binned_index rows of binned
        binned = binned[:last_binned_index, :, :]
        intensity = intensity[:last_binned_index, :, :]
        latent_coupling = latent_coupling[:last_binned_index, :]

        return intensity, binned, latent_coupling


    def sample_data(self):
        return self.binned, self.time

    def likelihood(self):
        intensity = self.intensity
        binned = self.binned
        dt = round(self.time[1] - self.time[0], 3)
        likelihood = np.sum(np.log(intensity) * binned - intensity * dt)
        return likelihood
