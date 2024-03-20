import numpy as np
from scipy.special import softmax
from scipy.stats import gamma
from src.EM_Torch.general_functions import softplus, inv_softplus

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
        self.trial_peak_offsets = None  # CR x 2AL

        # parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL

        # neuron parameters
        self.neuron_gains = None  # unobserved
        self.neuron_factor_assignments = None  # unobserved
        self.neuron_factor_assignments_onehot = None  # unobserved
        self.neuron_intensities = None  # unobserved

    def initialize(self, configs=3, A=2, T=200, intensity_type=('constant', '1peak', '2peaks'),
                   intensity_mltply=15, intensity_bias=1):
        degree = 3
        time = np.arange(0, T, 1) / 100
        self.time = time
        self.dt = round(self.time[1] - self.time[0], 3)

        n_factors = len(intensity_type) * A
        # paremeters
        self.alpha = inv_softplus(1*np.ones(n_factors, dtype=np.float64))
        self.theta = inv_softplus((1+np.arange(n_factors, dtype=np.float64))**(-1))
        self.pi = np.zeros(n_factors)
        self.config_peak_offsets = 0.01 * np.random.normal(size=(configs, 2 * n_factors))
        matrix = np.tril(np.random.normal(size=(2 * n_factors, 2 * n_factors)))
        cov = matrix @ matrix.T
        stdevs = np.sqrt(np.diag(cov))
        corr = cov / (stdevs[:, None] * stdevs[None, :])
        sdevs_scaled = stdevs / (max(np.max(stdevs), 2))
        # mask = np.tril(np.random.binomial(1, 0.9*np.ones_like(corr)), k=-1)
        # mask = mask + mask.T + np.eye(mask.shape[0])
        cov_scaled = corr * (sdevs_scaled[:, None] * sdevs_scaled[None, :])
        self.trial_peak_offset_covar_ltri = np.linalg.cholesky(cov_scaled)
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (np.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())
        # std_dev = np.sqrt(np.diag(trial_peak_offset_covar_matrix))
        # corr = np.diag(1/std_dev) @ trial_peak_offset_covar_matrix @ np.diag(1/std_dev)
        latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        latent_factors = np.vstack([latent_factors] * A)
        self.beta = np.log(latent_factors)
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

    def generate_neuron_gains_factor_assignments_condition_assignment_and_factor_access(self, n_neurons, n_areas):

        n_factors = self.beta.shape[0]
        n_configs = self.config_peak_offsets.shape[0]
        factors_per_area = n_factors//n_areas
        ratio = softmax(self.pi.reshape(n_areas, -1), axis=1)
        neuron_factor_access = np.zeros((n_configs, n_neurons, n_factors))
        neuron_factor_assignments = []
        neurons_assigned = 0
        neurons_per_area = n_neurons // n_areas
        for a in range(n_areas):
            area_start_indx = a * factors_per_area
            if a == n_areas-1:
                neurons_per_area = n_neurons - neurons_assigned
            neuron_factor_assignments.append(np.random.choice(factors_per_area, neurons_per_area*n_configs, p=ratio[a]).
                                              reshape(n_configs, -1) + area_start_indx)
            neurons_assigned += neurons_per_area
            neuron_factor_access[:, (a * neurons_per_area):((a + 1) * neurons_per_area), area_start_indx:(area_start_indx + factors_per_area)] = 1
        neuron_factor_assignments = np.concatenate(neuron_factor_assignments, axis=1)
        neuron_gains = np.random.gamma(softplus(self.alpha[neuron_factor_assignments]),
                                       softplus(self.theta[neuron_factor_assignments])**(-1))
        self.neuron_gains = neuron_gains
        self.neuron_factor_assignments = neuron_factor_assignments
        self.neuron_factor_access = neuron_factor_access

    def generate_spike_trains(self, trial_warped_factors):

        neuron_trial_warped_factors = []
        for i in range(self.neuron_factor_assignments.shape[0]):
            neuron_trial_warped_factors.append(trial_warped_factors[self.neuron_factor_assignments[i,:], :, :, i])
        neuron_trial_warped_factors = np.stack(neuron_trial_warped_factors, axis=3)
        neuron_intensities = self.neuron_gains.T[:,None,None,:] * neuron_trial_warped_factors
        # padding
        neuron_intensities = np.concatenate([np.zeros((neuron_intensities.shape[0], 1, neuron_intensities.shape[2], neuron_intensities.shape[3])), neuron_intensities], axis=1)
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
        Y = accepted_spikes * homogeneous_poisson_process
        self.Y = Y[:, 1:, :, :]
        self.neuron_intensities = neuron_intensities[:, 1:, :, :]


    def warp_all_latent_factors_for_all_trials(self, n_trials):
        n_factors = self.beta.shape[0]
        n_configs = self.config_peak_offsets.shape[0]
        trial_peak_offset_samples = (np.random.normal(0, 1, (1, n_trials * n_configs, 2 * n_factors))
                                     .reshape(1, n_trials, n_configs, 2 * n_factors))
        self.trial_peak_offsets = np.einsum('lj,nrcj->nrcl', self.trial_peak_offset_covar_ltri, trial_peak_offset_samples)
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks()
        warped_times = self.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors = self.compute_warped_factors(warped_times)
        return warped_factors


    def compute_offsets_and_landmarks(self):
        factors = np.exp(self.beta)
        avg_peak1_times = self.time[self.left_landmark1 + np.argmax(factors[:, self.left_landmark1:self.right_landmark1], axis=1)]
        avg_peak2_times = self.time[self.left_landmark2 + np.argmax(factors[:, self.left_landmark2:self.right_landmark2], axis=1)]
        avg_peak_times = np.hstack([avg_peak1_times, avg_peak2_times])
        # avg_peak_times  # 2AL
        # self.transformed_trial_peak_offset_samples  N x R x C x 2AL
        # self.config_peak_offsets  # C x 2AL
        offsets = self.trial_peak_offsets + np.expand_dims(self.config_peak_offsets, (0, 1))
        avg_peak_times = np.expand_dims(avg_peak_times, (0, 1, 2))
        s_new = avg_peak_times + offsets  # offset peak time
        left_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.left_landmark1, self.left_landmark2]), s_new.shape[-1] // 2), (0, 1, 2))]
        right_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.right_landmark1, self.right_landmark2]), s_new.shape[-1] // 2), (0, 1, 2))]
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
        # warped_time  # 50 x N x R X C X 2AL
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = np.exp(self.beta)
        # warped_time  # 50 x N x R X C X 2AL
        warped_indices = warped_times / self.dt
        floor_warped_indices = np.floor(warped_indices).astype(int)
        ceil_warped_indices = np.ceil(warped_indices).astype(int)
        ceil_weights = warped_indices - floor_warped_indices
        floor_weights = 1 - ceil_weights
        weighted_floor_warped_factors = []
        weighted_ceil_warped_factors = []
        for l in range(factors.shape[0]):
            floor_warped_factor_l = factors[l, floor_warped_indices[:, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[:, :, :, :, [l, (l + factors.shape[0])]]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[:, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[:, :, :, :, [l, (l + factors.shape[0])]]
            weighted_floor_warped_factors.append(weighted_floor_warped_factor_l)
            weighted_ceil_warped_factors.append(weighted_ceil_warped_factor_l)
        weighted_floor_warped_factors = np.stack(weighted_floor_warped_factors)
        weighted_ceil_warped_factors = np.stack(weighted_ceil_warped_factors)
        warped_factors = weighted_floor_warped_factors + weighted_ceil_warped_factors

        # warped_factors  # L x 50 x N x R X C X 2
        early = factors[:, :self.left_landmark1]
        early = np.broadcast_to(np.expand_dims(early, (2, 3, 4)), (early.shape) + tuple(warped_factors.shape[2:-1]))
        mid = factors[:, self.right_landmark1:self.left_landmark2]
        mid = np.broadcast_to(np.expand_dims(mid, (2, 3, 4)), (mid.shape) + tuple(warped_factors.shape[2:-1]))
        late = factors[:, self.right_landmark2:]
        late = np.broadcast_to(np.expand_dims(late, (2, 3, 4)), (late.shape) + tuple(warped_factors.shape[2:-1]))
        warped_factors = np.concatenate([early, warped_factors[:, :, :, :, :, 0], mid, warped_factors[:, :, :, :, :, 1], late], axis=1)
        return warped_factors


    def sample_data(self, K, A, n_trials):
        n_factors = self.beta.shape[0]
        n_configs = self.config_peak_offsets.shape[0]
        trial_warped_factors = self.warp_all_latent_factors_for_all_trials(n_trials).squeeze()
        self.generate_neuron_gains_factor_assignments_condition_assignment_and_factor_access(K, A)
        self.generate_spike_trains(trial_warped_factors)
        indcs = np.indices(self.neuron_factor_assignments.shape)
        neuron_factor_assignments = np.zeros((n_configs, K, n_factors))
        neuron_factor_assignments[indcs[0].flatten(), indcs[1].flatten(), self.neuron_factor_assignments.flatten()] = 1
        self.neuron_factor_assignments_onehot = neuron_factor_assignments
        self.neuron_factor_access = np.transpose(self.neuron_factor_access, (1, 2, 0))
        return self.Y, self.neuron_factor_access


    def get_sample_ground_truth(self):
        return (self.neuron_intensities, self.neuron_factor_assignments, self.neuron_factor_assignments_onehot,
                self.neuron_gains, self.trial_peak_offsets)

    # def compute_log_likelihood(self, Y, neuron_intensities, factor_assignment, config_offsets, trial_offset):
    #     ratio = softmax(self.pi, axis=0)[factor_assignment]
    #     likelihood = np.sum(np.log(neuron_intensities) * Y - neuron_intensities * self.dt)
    #     gain_prior = gamma.logpdf(self.neuron_gains, a=softplus(self.alpha)[factor_assignment],
    #                               scale=1/softplus(self.theta)[factor_assignment]).sum()
    #     membership_prior = np.sum(np.log(ratio))
    #     self.transformed_config_peak_offset_samples = config_offsets
    #     self.transformed_trial_peak_offset_samples = trial_offset
    #     offset_priors = self.compute_offset_entropy_terms()
    #     return likelihood+gain_prior+membership_prior+offset_priors
    #
    #
    # def compute_offset_entropy_terms(self):  # last 2 entropy terms
    #     # Entropy1 Terms
    #     dim = self.config_peak_offset_stdevs.shape[0]
    #
    #     Sigma1 = np.diag(softplus(self.config_peak_offset_stdevs)) @ np.diag(softplus(self.config_peak_offset_stdevs)).T
    #     det_Sigma1 = np.linalg.det(Sigma1)
    #     inv_Sigma1 = np.linalg.inv(Sigma1)
    #     prod_term1 = np.einsum('ncl,lj,ncj->nc', self.transformed_config_peak_offset_samples, inv_Sigma1, self.transformed_config_peak_offset_samples)  # sum over l
    #     # entropy_term1  # N x C
    #     entropy_term1 = -0.5 * np.sum(np.log((2 * np.pi) ** dim * det_Sigma1) + prod_term1)
    #
    #     Sigma2 = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
    #     det_Sigma2 = np.linalg.det(Sigma2)
    #     inv_Sigma2 = np.linalg.inv(Sigma2)
    #     prod_term2 = np.einsum('mrcl,lj,mrcj->mrc', self.transformed_trial_peak_offset_samples, inv_Sigma2, self.transformed_trial_peak_offset_samples)  # sum over l
    #     # entropy_term2  # M x C
    #     entropy_term2 = -0.5 * np.sum(np.log((2 * np.pi) ** dim * det_Sigma2) + prod_term2)
    #
    #     entropy_term = entropy_term1 + entropy_term2
    #     return entropy_term
