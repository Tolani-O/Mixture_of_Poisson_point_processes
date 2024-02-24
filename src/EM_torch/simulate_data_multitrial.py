import numpy as np
import torch
import torch.nn.functional as F
from src.EM_torch.model_data import ModelData

class DataAnalyzer(ModelData):

    def __init__(self):
        super().__init__()

        self.latent_factors = None  # unobserved
        self.neuron_gains = None  # unobserved
        self.neuron_factor_assignments = None  # unobserved
        self.neuron_intensities = None  # unobserved

    def initialize(self, A=2, K=100, T=200, intensity_type=('constant', '1peak', '2peaks'), intensity_mltply=15, intensity_bias=5):
        degree = 3
        time = np.arange(0, T, 1) / 100
        super().initialize(time, degree)
        n_factors = len(intensity_type) * A
        n_trials = 3
        n_configs = 2
        K = 5
        n_trial_samples = 1
        n_config_samples = 1
        # K is the number of neurons in a single condition across areas.
        # each condition has the same number of neurons, to total number of neurons across conditions is K * C
        self.randomly_initialize_parameters(n_factors, n_trials, n_configs, n_trial_samples, n_config_samples)
        self.latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        self.latent_factors = torch.from_numpy(np.vstack([self.latent_factors] * A))
        B_inv = torch.pinverse(self.B.to_dense().t())
        self.beta = torch.log(torch.matmul(self.latent_factors, B_inv.T))
        warped_factors, _, _ = self.warp_all_latent_factors_for_all_trials(False)
        trial_warped_factors = warped_factors.squeeze().detach().numpy()
        self.generate_neuron_gains_factor_assignments_condition_assignment_and_factor_access(K, n_configs, A)
        self.generate_spike_trains(trial_warped_factors)
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

    def generate_neuron_gains_factor_assignments_condition_assignment_and_factor_access(self, num_neurons, num_conditions, num_areas):

        num_factors_across_areas = len(self.pi)
        num_factors = num_factors_across_areas//num_areas
        ratio = F.softmax(self.pi, dim=0).detach().numpy()
        neuron_factor_assignments = np.random.choice(num_factors_across_areas, num_neurons*num_conditions, p=ratio).reshape(num_conditions, -1)
        neuron_factor_access = np.zeros((num_conditions, num_neurons, num_factors))
        for a in range(num_areas):
            neuron_factor_access[(((num_factors*a)<=neuron_factor_assignments)&((num_factors*a+2)>=neuron_factor_assignments)),:] = np.arange(3*a, 3*a+3)
        neuron_gains = np.random.gamma(self.alpha[neuron_factor_assignments.flatten()].detach().numpy(),
                                       self.theta[neuron_factor_assignments.flatten()].detach().numpy()).reshape(neuron_factor_assignments.shape)
        self.neuron_gains = neuron_gains
        self.neuron_factor_assignments = neuron_factor_assignments
        self.neuron_factor_access = neuron_factor_access

    def generate_spike_trains(self, trial_warped_factors):

        neuron_trial_warped_factors = []
        for i in range(self.neuron_factor_assignments.shape[0]):
            neuron_trial_warped_factors.append(trial_warped_factors[self.neuron_factor_assignments[i,:], :, :, i])
        neuron_trial_warped_factors = np.stack(neuron_trial_warped_factors, axis=3)
        neuron_intensities = self.neuron_gains.T[:,None,None,:] * neuron_trial_warped_factors
        rates = np.max(neuron_intensities, axis=1)
        arrival_times = np.zeros_like(rates)
        homogeneous_poisson_process = np.zeros_like(neuron_intensities)
        while np.min(arrival_times)<neuron_intensities.shape[1]:
            arrival_times += np.random.exponential(1/rates)
            update_entries = arrival_times < neuron_intensities.shape[1]
            update_indices = np.floor(update_entries * arrival_times).astype(int)
            dim_indices = np.indices(update_indices.shape)
            homogeneous_poisson_process[dim_indices[0].flatten(), update_indices.flatten(), dim_indices[1].flatten(), dim_indices[2].flatten()] = 1
        acceptance_threshold = neuron_intensities / rates[:, None, :, :]
        acceptance_probabilities = np.random.uniform(0, 1, neuron_intensities.shape)
        accepted_spikes = (acceptance_probabilities <= acceptance_threshold).astype(int)
        self.Y = accepted_spikes * homogeneous_poisson_process
        # self.Y[:, :, 0] = 0
        self.neuron_intensities = neuron_intensities

    def sample_data(self):
        return self.Y, self.time.numpy(), self.neuron_factor_access

    def compute_log_likelihood(self):
        likelihood = np.sum(np.log(self.neuron_intensities) * self.Y - self.neuron_intensities * self.dt)
        return likelihood
