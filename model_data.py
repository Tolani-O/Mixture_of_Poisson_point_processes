import numpy as np
from scipy.interpolate import BSpline


class ModelData:

    def __init__(self):
        self.time = None
        self.joint_factors_indices = None
        self.degree = None
        self.dt = None
        self.knots = None
        self.B = None
        self.Basis_elements = None
        landmark_spread = 50
        self.left_landmark1 = 20
        self.mid_landmark1 = self.left_landmark1 + landmark_spread/2
        self.right_landmark1 = self.left_landmark1 + landmark_spread
        self.left_landmark2 = 120
        self.mid_landmark2 = self.left_landmark2 + landmark_spread/2
        self.right_landmark2 = self.left_landmark2 + landmark_spread
        self.Y = None
        self.neuron_factor_access = None
        self.config_peak_offset_presamples = None  # C x 2AL
        self.trial_peak_offset_presamples = None  # CR x 2AL

        #parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.config_peak_offset_stdevs = None  # 2AL

    def initialize(self, time, degree=3):
        self.time = time
        self.degree = degree
        self.dt = round(self.time[1] - self.time[0], 3)
        self.knots = np.concatenate([np.repeat(self.time[0], degree), self.time, np.repeat(self.time[-1], degree)])
        self.knots[-1] = self.knots[-1] + self.dt
        self.B = BSpline.design_matrix(self.time, self.knots, degree).transpose()
        self.Basis_elements = []
        for p in range(len(self.knots) - degree - 1):
            if p < 2:
                times = np.array([p])
            elif p < self.left_landmark1:
                times = np.arange(start=p-2, stop=p+1)
            elif p < self.right_landmark1:
                times = np.arange(start=self.left_landmark1, stop=self.right_landmark1)
            elif p < self.left_landmark2:
                times = np.arange(start=p-2, stop=p+1)
            elif p < self.right_landmark2:
                times = np.arange(start=self.left_landmark2, stop=self.right_landmark2)
            elif p < len(self.time):
                times = np.arange(start=p-2, stop=p+1)
            else:
                times = np.array([p-2])
            basis_p = BSpline.basis_element(self.knots[p:(p + degree + 2)], extrapolate=False)
            self.Basis_elements.append((basis_p, times))

    def randomly_initialize_parameters(self, n_factors=6, n_trials=15, n_configs=40, n_trial_samples=11, n_config_samples=9):
        # fixed values
        self.config_peak_offset_presamples = np.random.normal(0, 1, (n_config_samples, n_configs, 2 * n_factors))
        self.trial_peak_offset_presamples = np.random.normal(0, 1, (n_trial_samples, n_trials * n_configs, 2 * n_factors)).reshape(n_trial_samples, n_trials, n_configs, 2 * n_factors)

        #paremeters
        self.beta = np.random.rand(n_factors, self.B.shape[0])
        self.alpha = np.random.rand(n_factors)
        self.theta = np.random.rand(n_factors)
        self.pi = np.random.rand(n_factors)
        self.config_peak_offset_stdevs = np.random.rand(2 * n_factors)
        bounds = 0.05
        matrix = np.tril(np.random.uniform(-bounds, bounds, (2 * n_factors, 2 * n_factors)))
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

    def warp_all_latent_factors_for_all_trials(self, gamma, derivs=True):

        transformed_trial_peak_offsets = np.einsum('lj,mrcj->mrcl', self.trial_peak_offset_covar_ltri, self.trial_peak_offset_presamples)
        transformed_config_peak_offsets = np.einsum('l,ncl->ncl', self.config_peak_offset_stdevs, self.config_peak_offset_presamples)
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks(gamma, transformed_config_peak_offsets, transformed_trial_peak_offsets)
        warped_times = self.warp_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors, sum_RT_Y_times_warped_bases, sum_RT_warped_factors_times_warped_bases = self.recompute_warp_terms(gamma, warped_times, derivs)
        return warped_factors, warped_times, sum_RT_Y_times_warped_bases, sum_RT_warped_factors_times_warped_bases, transformed_trial_peak_offsets, transformed_config_peak_offsets

    def compute_offsets_and_landmarks(self, gamma, transformed_config_peak_offsets, transformed_trial_peak_offsets):

        factors = gamma @ self.B
        avg_peak1_times = self.time[self.left_landmark1 + np.argmax(factors[:, self.left_landmark1:self.right_landmark1], axis=1)]
        avg_peak2_times = self.time[self.left_landmark2 + np.argmax(factors[:, self.left_landmark2:self.right_landmark2], axis=1)]
        avg_peak_times = np.hstack([avg_peak1_times, avg_peak2_times])
        # offsets  # M X N x R X C X L
        offsets = np.expand_dims(transformed_trial_peak_offsets, 1) + np.expand_dims(transformed_config_peak_offsets, (0, 2))
        avg_peak_times = np.expand_dims(avg_peak_times, (0, 1, 2, 3))
        s_new = avg_peak_times + offsets  # offset peak time
        left_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.left_landmark1, self.left_landmark2]), s_new.shape[-1] // 2), (0, 1, 2, 3))]
        right_landmarks = self.time[np.expand_dims(np.repeat(np.array([self.right_landmark1, self.right_landmark2]), s_new.shape[-1] // 2), (0, 1, 2, 3))]
        s_new = np.where(s_new <= left_landmarks, left_landmarks + self.dt, s_new)
        s_new = np.where(s_new >= right_landmarks, right_landmarks - self.dt, s_new)
        return avg_peak_times, left_landmarks, right_landmarks, s_new

    def warp_times(self, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        landmark_spead = self.right_landmark1 - self.left_landmark1
        left_shifted_time = np.arange(0, self.time[landmark_spead], self.dt)
        left_shifted_peak_times = trial_peak_times - left_landmarks
        right_shifted_peak_times = trial_peak_times - right_landmarks
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        warped_time = np.stack([np.zeros_like(trial_peak_times)] * left_shifted_time.shape[0])
        for i in range(left_shifted_time.shape[0]):
            warped_time[i] = np.where(left_shifted_time[i] < left_shifted_peak_times, (left_shifted_time[i]*left_slope)+left_landmarks,
                                         ((left_shifted_time[i]-left_shifted_peak_times)*right_slope)+avg_peak_times)
        early = self.time[:self.left_landmark1]
        early = np.broadcast_to(np.expand_dims(early, (1,2,3,4,5)), (early.shape) + tuple(warped_time.shape[1:-1]) + (6,))
        mid = self.time[self.right_landmark1:self.left_landmark2]
        mid = np.broadcast_to(np.expand_dims(mid, (1,2,3,4,5)), (mid.shape) + tuple(warped_time.shape[1:-1]) + (6,))
        late = self.time[self.right_landmark2:]
        late = np.broadcast_to(np.expand_dims(late, (1,2,3,4,5)), (late.shape) + tuple(warped_time.shape[1:-1]) + (6,))
        # warped_time  # T x M X N x R X C X L
        warped_times = np.vstack([early, warped_time[:,:,:,:,:,:6], mid, warped_time[:,:,:,:,:,6:], late])
        return warped_times


    def recompute_warp_terms(self, gamma, warped_time, derivs=False):
        warped_factors = []
        sum_RT_Y_times_warped_bases = []
        sum_RT_warped_factors_times_warped_bases = []
        for l in range(gamma.shape[0]):
            l_warped_time = warped_time[:, :, :, :, :, l]
            spl = BSpline(self.knots, gamma[l].T, self.degree)
            warped_factor_l = spl(l_warped_time)
            warped_factors.append(warped_factor_l)
            if derivs:
                factor_bases = []
                Y_bases = []
                for p in range(len(self.Basis_elements)):
                    p_object = self.Basis_elements[p]
                    spline = p_object[0]
                    evaluate_at = p_object[1]
                    if len(evaluate_at) <= self.degree:
                        points = self.time[evaluate_at]
                        basis_p = np.broadcast_to(np.expand_dims(spline(points), axis=(1, 2, 3, 4)), [evaluate_at.shape[0]] + list(l_warped_time.shape[1:]))
                    else:
                        points = l_warped_time[evaluate_at]
                        basis_p = np.nan_to_num(spline(points.flatten())).reshape(points.shape)
                    sum_RT_Y_times_warped_bases_item = np.einsum('ktrc,tmnrc->kmnc', self.Y[:, evaluate_at, :, :], basis_p)
                    Y_bases.append(sum_RT_Y_times_warped_bases_item)
                    sum_RT_warped_factors_times_warped_bases_item = np.einsum('tmnrc,tmnrc->mnc', warped_factor_l[evaluate_at, :, :, :, :], basis_p)
                    factor_bases.append(sum_RT_warped_factors_times_warped_bases_item)
                sum_RT_Y_times_warped_bases.append(np.stack(Y_bases))
                sum_RT_warped_factors_times_warped_bases.append(np.stack(factor_bases))
        warped_factors = np.stack(warped_factors)
        if derivs:
            sum_RT_Y_times_warped_bases = np.stack(sum_RT_Y_times_warped_bases)
            sum_RT_warped_factors_times_warped_bases = np.stack(sum_RT_warped_factors_times_warped_bases)
        return warped_factors, sum_RT_Y_times_warped_bases, sum_RT_warped_factors_times_warped_bases
