import numpy as np
import torch

from src.EM_torch.model_data import ModelData

class DataAnalyzer(ModelData):

    def __init__(self):
        super().__init__()
        # self.time = None
        # self.joint_factors_indices = None
        # self.degree = None
        # self.dt = None
        # self.knots = None
        # self.B = None
        # self.left_landmark1 = 20
        # self.mid_landmark1 = 45
        # self.right_landmark1 = 70
        # self.left_landmark2 = 120
        # self.mid_landmark2 = 145
        # self.right_landmark2 = 170
        # self.Y = None
        # self.trial_warped_factors = None
        # self.trial_warped_splines = None
        # self.trial_peak_offsets = None
        #
        # # parameters
        # self.beta = None  # AL x P
        # self.alpha = None  # 1 x AL
        # self.theta = None  # 1 x AL
        # self.trial_offset_covar_matrix = None

        self.latent_factors = None
        self.neuron_gains = None
        self.neuron_factor_assignments = None
        self.neuron_intensities = None

    def initialize(self, joint_factor_indices, degree=3, A=2, K=100, R=50, T=200,
                   intensity_type=('constant', '1peak', '2peaks'), ratio=1, intensity_mltply=15, intensity_bias=5):
        time = np.arange(0, T, 1) / 100
        super().initialize(time, joint_factor_indices, degree)

        self.latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        self.latent_factors = torch.from_numpy(np.vstack([self.latent_factors] * A))
        B_inv = torch.pinverse(self.B.to_dense().t())
        self.beta = torch.log(torch.matmul(self.latent_factors, B_inv.T))

        self.randomly_initialize_parameters()

        L = self.latent_factors.shape[0]
        warp = 0.04
        covar_matrix = np.random.uniform(-warp, warp, (2*L, 2*L))
        self.trial_offset_covar_matrix = covar_matrix.T @ covar_matrix
        # compute trial_warped_factors and trial_warped_splines

        self.warp_all_latent_factors_for_all_trials()
        self.construct_weight_matrices()

        # self.randomly_initialize_parameters()
        self.generate_neuron_gains_and_factor_assignments(K, L, ratio)
        self.generate_spike_trains()

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

    def generate_neuron_gains_and_factor_assignments(self, num_neurons, num_factors, ratio):
        if num_factors == 1:
            ratio = [1]
        elif isinstance(ratio, int):  # if ratio is 1 in this case, it will be normalized in a few lines,
            # and amounts to a uniform distribution of neurons
            ratio = [ratio] * num_factors
        # make sure num_factors and len(ratio) are the same
        assert num_factors == len(ratio)
        # make sure ratio sums to 1
        ratio = np.array(ratio)
        ratio = ratio / np.sum(ratio)

        neuron_gains = np.zeros(num_neurons)
        neuron_factor_assignments = np.zeros(num_neurons)
        last_index = 0
        for l in range(num_factors):
            neuron_count = int(num_neurons * ratio[l])
            neuron_gains[last_index:(last_index+neuron_count)] = np.random.gamma(self.alpha[l], self.theta[l], neuron_count)
            neuron_factor_assignments[last_index:(last_index+neuron_count)] = l
            last_index += neuron_count
        self.neuron_gains = neuron_gains
        self.neuron_factor_assignments = neuron_factor_assignments.astype(int)
        self.pi = ratio

    def generate_spike_trains(self):

        neuron_intensities = self.neuron_gains[:, np.newaxis, np.newaxis] * self.trial_warped_factors[self.neuron_factor_assignments, :, :]
        rates = np.max(neuron_intensities, axis=2)
        arrival_times = np.zeros_like(rates)
        homogeneous_poisson_process = np.zeros_like(neuron_intensities)
        while np.min(arrival_times)<neuron_intensities.shape[2]:
            arrival_times += np.random.exponential(1/rates)
            update_entries = arrival_times < neuron_intensities.shape[2]
            update_indices = np.floor(update_entries * arrival_times).astype(int)
            row_indices, col_indices = np.indices(update_indices.shape)
            update_indices_flat = update_indices.flatten()
            row_indices_flat = row_indices.flatten()
            col_indices_flat = col_indices.flatten()
            homogeneous_poisson_process[row_indices_flat, col_indices_flat, update_indices_flat] = 1
        acceptance_threshold = neuron_intensities / rates[:, :, np.newaxis]
        acceptance_probabilities = np.random.uniform(0, 1, neuron_intensities.shape)
        accepted_spikes = (acceptance_probabilities <= acceptance_threshold).astype(int)
        self.Y = accepted_spikes * homogeneous_poisson_process
        self.Y[:, :, 0] = 0
        self.neuron_intensities = neuron_intensities

    def sample_data(self):
        return self.Y, self.time

    def compute_log_likelihood(self):
        likelihood = np.sum(np.log(self.neuron_intensities) * self.Y - self.neuron_intensities * self.dt)
        return likelihood
