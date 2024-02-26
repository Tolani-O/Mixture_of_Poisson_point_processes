import numpy as np
from scipy.special import softmax
from scipy.interpolate import BSpline

class DataAnalyzer:

    def __init__(self):
        self.time = None
        self.dt = None
        landmark_spread = 50
        self.left_landmark1 = 20
        self.mid_landmark1 = self.left_landmark1 + landmark_spread / 2
        self.right_landmark1 = self.left_landmark1 + landmark_spread
        self.left_landmark2 = 120
        self.mid_landmark2 = self.left_landmark2 + landmark_spread / 2
        self.right_landmark2 = self.left_landmark2 + landmark_spread
        self.Y = None
        self.neuron_factor_access = None
        self.config_peak_offset_presamples = None  # C x 2AL
        self.trial_peak_offset_presamples = None  # CR x 2AL
        self.transformed_trial_peak_offset_samples = None  # CR x 2AL
        self.transformed_config_peak_offset_samples = None  # C x 2AL

        # parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.config_peak_offset_stdevs = None  # 2AL

        # neuron parameters
        self.latent_factors = None  # unobserved
        self.neuron_gains = None  # unobserved
        self.neuron_factor_assignments = None  # unobserved
        self.neuron_intensities = None  # unobserved

    def initialize(self, A=2, K=100, T=200, intensity_type=('constant', '1peak', '2peaks'),
                   intensity_mltply=15, intensity_bias=5, n_trials=3, n_configs=2):
        degree = 3
        time = np.arange(0, T, 1) / 100
        self.time = time
        self.dt = round(self.time[1] - self.time[0], 3)

        n_factors = len(intensity_type) * A
        # K is the number of neurons in a single condition across areas.
        # each condition has the same number of neurons, to total number of neurons across conditions is K * C
        # fixed values
        self.config_peak_offset_presamples = np.random.normal(0, 1, (1, n_configs, 2 * n_factors))
        self.trial_peak_offset_presamples = (np.random.normal(0, 1, (1, n_trials * n_configs, 2 * n_factors))
                                             .reshape(1, n_trials, n_configs, 2 * n_factors))
        # paremeters
        self.beta = np.random.normal(size=(n_factors, self.time.shape[0]))
        self.alpha = np.random.normal(size=n_factors)
        self.theta = np.random.normal(size=n_factors)
        self.pi = np.random.normal(size=n_factors-1)
        self.config_peak_offset_stdevs = np.random.normal(size=2 * n_factors)
        bounds = 0.05
        matrix = np.tril(np.random.normal(size=(2 * n_factors, 2 * n_factors)))
        # Ensure diagonal elements are positive
        for i in range(min(matrix.shape)):
            matrix[i, i] += np.exp(matrix[i, i])
        # Make it a learnable parameter
        self.trial_peak_offset_covar_ltri = matrix
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (np.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())
        # std_dev = np.sqrt(np.diag(trial_peak_offset_covar_matrix))
        # corr = np.diag(1/std_dev) @ trial_peak_offset_covar_matrix @ np.diag(1/std_dev)

        latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        latent_factors = np.vstack([latent_factors] * A)
        self.beta = np.log(latent_factors)
        warped_factors = self.warp_all_latent_factors_for_all_trials(np.exp(self.beta))
        trial_warped_factors = warped_factors.squeeze()
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

        num_factors = len(self.pi)+1
        ratio = softmax(np.hstack([np.zeros(1), self.pi]), axis=0)
        neuron_factor_assignments = np.random.choice(num_factors, num_neurons*num_conditions, p=ratio).reshape(num_conditions, -1)
        neuron_factor_access = np.zeros((num_conditions, num_neurons, num_factors))
        for a in range(num_areas):
            area_start_indx = a * (num_factors//2)
            neuron_factor_access[((area_start_indx<=neuron_factor_assignments)&((area_start_indx+2)>=neuron_factor_assignments)), (3*a):(3*a+3)] = 1
        neuron_gains = (np.random.gamma(np.exp(self.alpha[neuron_factor_assignments.flatten()]),
                                       np.exp(self.theta[neuron_factor_assignments.flatten()]))
                        .reshape(neuron_factor_assignments.shape))
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


    def warp_all_latent_factors_for_all_trials(self, gamma):

        self.transformed_trial_peak_offsets = np.einsum('lj,mrcj->mrcl', self.trial_peak_offset_covar_ltri, self.trial_peak_offset_presamples)
        self.transformed_config_peak_offsets = np.einsum('l,ncl->ncl', np.exp(self.config_peak_offset_stdevs), self.config_peak_offset_presamples)
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks()
        warped_times = self.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors = self.compute_warped_factors(warped_times)
        return warped_factors

    def compute_offsets_and_landmarks(self):

        factors = np.exp(self.beta)
        avg_peak1_times = self.time[self.left_landmark1 + np.argmax(factors[:, self.left_landmark1:self.right_landmark1], axis=1)]
        avg_peak2_times = self.time[self.left_landmark2 + np.argmax(factors[:, self.left_landmark2:self.right_landmark2], axis=1)]
        avg_peak_times = np.hstack([avg_peak1_times, avg_peak2_times])
        # offsets  # M X N x R X C X L
        offsets = np.expand_dims(self.transformed_trial_peak_offsets, 1) + np.expand_dims(self.transformed_config_peak_offsets, (0, 2))
        avg_peak_times = np.expand_dims(avg_peak_times, (0, 1, 2, 3))
        s_new = avg_peak_times + offsets  # offset peak time
        left_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.left_landmark1, self.left_landmark2]), s_new.shape[-1] // 2), (0, 1, 2, 3))]
        right_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.right_landmark1, self.right_landmark2]), s_new.shape[-1] // 2), (0, 1, 2, 3))]
        s_new = np.where(s_new <= left_landmarks, left_landmarks + self.dt, s_new)
        s_new = np.where(s_new >= right_landmarks, right_landmarks - self.dt, s_new)
        return avg_peak_times, left_landmarks, right_landmarks, s_new

    def compute_warped_times(self, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        landmark_spead = self.right_landmark1 - self.left_landmark1
        left_shifted_time = np.arange(0, self.time[landmark_spead], self.dt)
        left_shifted_peak_times = trial_peak_times - left_landmarks
        right_shifted_peak_times = trial_peak_times - right_landmarks
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        warped_times = np.stack([np.zeros_like(trial_peak_times)] * left_shifted_time.shape[0])
        for i in range(left_shifted_time.shape[0]):
            warped_times[i] = np.where(left_shifted_time[i] < left_shifted_peak_times, (left_shifted_time[i]*left_slope)+left_landmarks,
                                         ((left_shifted_time[i]-left_shifted_peak_times)*right_slope)+avg_peak_times)
        # landmark_spead = 50
        # warped_time  # 50 x M X N x R X C X 2L
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = np.exp(self.beta)
        # warped_time  # 50 x M X N x R X C X 2L
        warped_indices = warped_times / self.dt
        floor_warped_indices = np.floor(warped_indices).astype(int)
        ceil_warped_indices = np.ceil(warped_indices).astype(int)
        ceil_weights = warped_indices - floor_warped_indices
        floor_weights = 1 - ceil_weights
        weighted_floor_warped_factors = []
        weighted_ceil_warped_factors = []
        for l in range(factors.shape[0]):
            floor_warped_factor_l = factors[l, floor_warped_indices[:, :, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[:, :, :, :, :,
                                                                     [l, (l + factors.shape[0])]]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[:, :, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[:, :, :, :, :,
                                                                   [l, (l + factors.shape[0])]]
            weighted_floor_warped_factors.append(weighted_floor_warped_factor_l)
            weighted_ceil_warped_factors.append(weighted_ceil_warped_factor_l)
        weighted_floor_warped_factors = np.stack(weighted_floor_warped_factors)
        weighted_ceil_warped_factors = np.stack(weighted_ceil_warped_factors)
        warped_factors = weighted_floor_warped_factors + weighted_ceil_warped_factors

        early = factors[:, :self.left_landmark1]
        early = np.broadcast_to(np.expand_dims(early, (2, 3, 4, 5)), (early.shape) + tuple(warped_factors.shape[2:-1]))
        mid = factors[:, self.right_landmark1:self.left_landmark2]
        mid = np.broadcast_to(np.expand_dims(mid, (2, 3, 4, 5)), (mid.shape) + tuple(warped_factors.shape[2:-1]))
        late = factors[:, self.right_landmark2:]
        late = np.broadcast_to(np.expand_dims(late, (2, 3, 4, 5)), (late.shape) + tuple(warped_factors.shape[2:-1]))
        warped_factors = np.concatenate([early, warped_factors[:, :, :, :, :, :, 0], mid, warped_factors[:, :, :, :, :, :, 1], late], axis=1)

        return warped_factors


    def sample_data(self):
        return self.Y, self.time, self.neuron_factor_access

    def compute_log_likelihood(self):
        likelihood = np.sum(np.log(self.neuron_intensities) * self.Y - self.neuron_intensities * self.dt)
        return likelihood
