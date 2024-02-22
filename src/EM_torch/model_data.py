import numpy as np
from scipy.interpolate import BSpline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


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
        self.trial_warped_factors = None
        self.config_peak_offset_presamples = None  # CR x 2AL
        self.trial_peak_offset_presamples = None  # CR x 2AL

        #parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.config_peak_offset_stdevs = None  # 2AL

    def initialize(self, time, joint_factor_indices, degree=3):
        self.time = torch.from_numpy(time)
        self.joint_factors_indices = joint_factor_indices
        self.degree = degree
        self.dt = torch.round((self.time[1] - self.time[0]) * 1000) / 1000
        self.knots = np.concatenate([np.repeat(self.time[0], degree), self.time, np.repeat(self.time[-1], degree)])
        self.knots[-1] = self.knots[-1] + self.dt
        self.B = torch.from_numpy(BSpline.design_matrix(self.time, self.knots, degree).transpose().toarray()).to_sparse_coo()
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

    def randomly_initialize_parameters(self, n_trials=15, n_configs=40, n_trial_samples=15, n_config_samples=5):
        # fixed values
        presamples = torch.normal(0, 1, (n_config_samples, n_configs, 2 * self.beta.shape[0]))
        self.config_peak_offset_presamples = presamples.repeat_interleave(n_trial_samples, dim=1)
        self.trial_peak_offset_presamples = torch.normal(0, 1, (n_trial_samples, n_trials * n_configs, 2 * self.beta.shape[0]))

        #paremeters
        self.alpha = torch.nn.Parameter(torch.rand(self.beta.shape[0]))
        self.theta = torch.nn.Parameter(torch.rand(self.beta.shape[0]))
        self.pi = torch.nn.Parameter(torch.rand(self.beta.shape[0]))
        self.config_peak_offset_stdevs = torch.nn.Parameter(torch.rand(2 * self.beta.shape[0]))
        bounds = 0.05
        matrix = torch.tril(torch.empty((2 * self.beta.shape[0], 2 * self.beta.shape[0])).uniform_(-bounds, bounds))
        matrix.diagonal().add_(0.1)  # Ensure diagonal elements are positive
        # Make it a learnable parameter
        self.trial_peak_offset_covar_ltri = torch.nn.Parameter(matrix)

    def construct_weight_matrices(self):
        warped_factors, warped_bases = self.warp_all_latent_factors_for_all_trials()
        a = 1

    def warp_all_latent_factors_for_all_trials(self):
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = torch.mm(self.trial_peak_offset_covar_matrix, self.trial_peak_offset_covar_matrix.t())
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (torch.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())

        transformed_trial_peak_offsets = torch.einsum('ij,klj->kli', self.trial_peak_offset_covar_ltri, self.trial_peak_offset_presamples)
        transformed_config_peak_offsets = torch.einsum('j,klj->klj', self.config_peak_offset_stdevs, self.config_peak_offset_presamples)
        factors = torch.matmul(torch.exp(self.beta), self.B)
        avg_peak1_times = self.time[self.left_landmark1 + torch.argmax(factors[:, self.left_landmark1:self.right_landmark1], dim=1)]
        avg_peak2_times = self.time[self.left_landmark2 + torch.argmax(factors[:, self.left_landmark2:self.right_landmark2], dim=1)]
        avg_peak_times = torch.cat([avg_peak1_times, avg_peak2_times])
        offsets = torch.add(transformed_trial_peak_offsets.unsqueeze(1), transformed_config_peak_offsets.unsqueeze(0))
        avg_peak_times = avg_peak_times.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        s_new = torch.add(avg_peak_times, offsets) # offset peak time
        left_landmarks = self.time[torch.cat([torch.tensor([self.left_landmark1]), torch.tensor([self.left_landmark2])]).repeat_interleave(s_new.shape[-1] // 2).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(s_new)]
        right_landmarks = self.time[torch.cat([torch.tensor([self.right_landmark1]), torch.tensor([self.right_landmark2])]).repeat_interleave(s_new.shape[-1] // 2).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(s_new)]
        s_new = torch.where(s_new <= left_landmarks, left_landmarks+self.dt, s_new)
        s_new = torch.where(s_new >= right_landmarks, right_landmarks-self.dt, s_new)
        warped_factors, warped_bases = self.warp_latent_factors(torch.exp(self.beta.detach()), avg_peak_times, left_landmarks, right_landmarks, s_new)
        return warped_factors, warped_bases

    def warp_latent_factors(self, gamma, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        landmark_spead = self.right_landmark1 - self.left_landmark1
        left_shifted_time = torch.arange(0, self.time[landmark_spead], self.dt)
        left_shifted_peak_times = trial_peak_times - left_landmarks
        right_shifted_peak_times = trial_peak_times - right_landmarks
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        warped_time = torch.stack([torch.zeros_like(trial_peak_times)] * left_shifted_time.shape[0])
        for i in range(left_shifted_time.shape[0]):
            warped_time[i] = torch.where(left_shifted_time[i] < left_shifted_peak_times, (left_shifted_time[i]*left_slope)+left_landmarks,
                                         ((left_shifted_time[i]-left_shifted_peak_times)*right_slope)+avg_peak_times)
        early = self.time[:self.left_landmark1]
        early = early.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((-1,) + tuple(warped_time.shape[1:-1]) + (6,))
        mid = self.time[self.right_landmark1:self.left_landmark2]
        mid = mid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((-1,) + tuple(warped_time.shape[1:-1]) + (6,))
        late = self.time[self.right_landmark2:]
        late = late.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((-1,) + tuple(warped_time.shape[1:-1]) + (6,))
        warped_time = torch.cat([early, warped_time[:,:,:,:,:6], mid, warped_time[:,:,:,:,6:], late], dim=0)
        warped_factors = []
        warped_bases = []
        for l in range(gamma.shape[0]):
            l_warped_time = warped_time[:,:,:,:,l].detach().numpy()
            spl = BSpline(self.knots, gamma[l].t(), self.degree)
            warped_factors.append(spl(l_warped_time))
            bases = []
            for p in range(len(self.Basis_elements)):
                p_object = self.Basis_elements[p]
                spline = p_object[0]
                evaluate_at = p_object[1]
                if len(evaluate_at) <=self.degree:
                    points = self.time[evaluate_at]
                    basis_p = np.broadcast_to(spline(points)[:,None,None,None], np.array([evaluate_at.shape[0]]+list(l_warped_time.shape[1:])))
                else:
                    points = l_warped_time[evaluate_at]
                    basis_p = np.nan_to_num(spline(points.flatten())).reshape(points.shape)
                bases.append(basis_p)
            warped_bases.append(bases)
        warped_factors = torch.from_numpy(np.stack(warped_factors))
        return warped_factors, warped_bases

    def compute_log_likelihood(self):
        pass
