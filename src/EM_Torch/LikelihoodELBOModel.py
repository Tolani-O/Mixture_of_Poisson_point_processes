import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix, inv_softplus
import numpy as np
from scipy.ndimage import gaussian_filter1d


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_trial_samples, n_config_samples):
        super(LikelihoodELBOModel, self).__init__()

        self.time = torch.tensor(time)
        dt = round(time[1] - time[0], 3)
        self.dt = torch.tensor(dt)
        T = time.shape[0]
        landmark_spread = 50
        self.left_landmark1 = 20
        self.mid_landmark1 = self.left_landmark1 + landmark_spread / 2
        self.right_landmark1 = self.left_landmark1 + landmark_spread
        self.left_landmark2 = 120
        self.mid_landmark2 = self.left_landmark2 + landmark_spread / 2
        self.right_landmark2 = self.left_landmark2 + landmark_spread
        self.neuron_factor_access = None
        self.transformed_trial_peak_offset_samples = None  # MCR x 2AL
        self.transformed_config_peak_offset_samples = None  # NC x 2AL
        self.n_config_samples = n_config_samples
        self.n_trial_samples = n_trial_samples
        self.W_C_tensor = None  # 1 x 1 x M x N x C
        Delta2 = create_second_diff_matrix(T)
        self.Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2).to_sparse()  # T x T # tikhonov regularization

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offset_stdevs = None  # 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.smoothness_budget = None  # L x 1


    def init_random(self, n_factors):
        self.beta = nn.Parameter(torch.randn(n_factors, self.time.shape[0]))
        self.alpha = nn.Parameter(torch.randn(n_factors))
        self.theta = nn.Parameter(torch.randn(n_factors))
        self.pi = nn.Parameter(torch.randn(n_factors-1))
        self.config_peak_offset_stdevs = nn.Parameter(torch.randn(2 * n_factors))
        matrix = torch.tril(torch.randn(2 * n_factors, 2 * n_factors))
        # Ensure diagonal elements are positive
        for i in range(min(matrix.size())):
            matrix[i, i] += F.softplus(matrix[i, i])
        # Make it a learnable parameter
        self.trial_peak_offset_covar_ltri = nn.Parameter(matrix)
        self.smoothness_budget = nn.Parameter(torch.zeros(n_factors-1, dtype=torch.float64))
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (np.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())
        # std_dev = np.sqrt(np.diag(trial_peak_offset_covar_matrix))
        # corr = np.diag(1/std_dev) @ trial_peak_offset_covar_matrix @ np.diag(1/std_dev)


    def init_ground_truth(self, n_factors, beta=None, alpha=None, theta=None, pi=None,
                          config_peak_offset_stdevs=None, trial_peak_offset_covar_ltri=None):
        self.init_random(n_factors)
        if beta is not None:
            self.beta = nn.Parameter(beta)
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
        if theta is not None:
            self.theta = nn.Parameter(theta)
        if pi is not None:
            self.pi = nn.Parameter(pi)
        if config_peak_offset_stdevs is not None:
            self.config_peak_offset_stdevs = nn.Parameter(config_peak_offset_stdevs)
        if trial_peak_offset_covar_ltri is not None:
            self.trial_peak_offset_covar_ltri = nn.Parameter(trial_peak_offset_covar_ltri)


    def init_from_data(self, Y, neuron_factor_access, factor_indcs):
        # Y # K x T x R x C
        # neuron_factor_access  # C x K x L
        K, T, R, C = Y.shape
        n_factors = neuron_factor_access.shape[-1]
        averaged_neurons = np.einsum('ktrc,ckl->tl', Y, neuron_factor_access) / (K*C*R)
        latent_factors = np.apply_along_axis(gaussian_filter1d, axis=0, arr=averaged_neurons, sigma=4).T
        beta = torch.zeros((n_factors, T))
        beta[factor_indcs] = torch.tensor(inv_softplus(latent_factors[factor_indcs]))
        props = torch.sum(neuron_factor_access, dim=(0, 1)) / (K*C)
        # true_props = props/torch.sum(props)
        tr_props = torch.log(props) + torch.logsumexp(props, dim=0)
        tr_props = tr_props - tr_props[0]
        pi = tr_props[1:]
        self.init_ground_truth(n_factors, beta=beta, pi=pi)


    def warp_all_latent_factors_for_all_trials(self, n_configs, n_trials):
        n_factors = self.beta.shape[0]
        config_peak_offset_samples = torch.randn(self.n_config_samples, n_configs, 2 * n_factors, dtype=torch.float64)
        trial_peak_offset_samples = torch.randn(self.n_trial_samples, n_trials * n_configs, 2 * n_factors, dtype=torch.float64).view(
            self.n_trial_samples, n_trials, n_configs, 2 * n_factors)
        self.transformed_trial_peak_offset_samples = torch.einsum('lj,mrcj->mrcl', self.trial_peak_offset_covar_ltri, trial_peak_offset_samples)
        self.transformed_config_peak_offset_samples = torch.einsum('l,ncl->ncl', F.softplus(self.config_peak_offset_stdevs), config_peak_offset_samples)
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks()
        warped_times = self.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors = self.compute_warped_factors(warped_times)
        return warped_factors


    def compute_offsets_and_landmarks(self):
        factors = F.softplus(self.beta)
        avg_peak1_times = self.time[self.left_landmark1 + torch.argmax(factors[:, self.left_landmark1:self.right_landmark1], dim=1)]
        avg_peak2_times = self.time[self.left_landmark2 + torch.argmax(factors[:, self.left_landmark2:self.right_landmark2], dim=1)]
        avg_peak_times = torch.cat([avg_peak1_times, avg_peak2_times])
        offsets = self.transformed_trial_peak_offset_samples.unsqueeze(1) + self.transformed_config_peak_offset_samples.unsqueeze(0).unsqueeze(2)
        avg_peak_times = avg_peak_times.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        s_new = avg_peak_times + offsets
        left_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.left_landmark1, self.left_landmark2]), s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        right_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.right_landmark1, self.right_landmark2]), s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        s_new = torch.where(s_new <= left_landmarks, left_landmarks + self.dt, s_new)
        s_new = torch.where(s_new >= right_landmarks, right_landmarks - self.dt, s_new)
        return avg_peak_times, left_landmarks, right_landmarks, s_new


    def compute_warped_times(self, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        landmark_spead = self.right_landmark1 - self.left_landmark1
        left_shifted_time = torch.arange(0, self.time[landmark_spead], self.dt)
        left_shifted_peak_times = trial_peak_times - left_landmarks
        right_shifted_peak_times = trial_peak_times - right_landmarks
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        warped_times = torch.stack([torch.zeros_like(trial_peak_times)] * left_shifted_time.shape[0])
        for i in range(left_shifted_time.shape[0]):
            warped_times[i] = torch.where(left_shifted_time[i] < left_shifted_peak_times, (left_shifted_time[i] * left_slope) + left_landmarks,
                                         ((left_shifted_time[i] - left_shifted_peak_times) * right_slope) + avg_peak_times)
        # landmark_spead = 50
        # warped_time  # 50 x M X N x R X C X 2L
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = F.softplus(self.beta)
        # warped_time  # 50 x M X N x R X C X 2L
        warped_indices = warped_times / self.dt
        floor_warped_indices = torch.floor(warped_indices).int()
        ceil_warped_indices = torch.ceil(warped_indices).int()
        ceil_weights = warped_indices - floor_warped_indices
        floor_weights = 1 - ceil_weights
        weighted_floor_warped_factors = []
        weighted_ceil_warped_factors = []
        for l in range(factors.shape[0]):
            floor_warped_factor_l = factors[l, floor_warped_indices[:,:,:,:,:,[l,(l+factors.shape[0])]]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[:,:,:,:,:,[l,(l+factors.shape[0])]]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[:,:,:,:,:,[l,(l+factors.shape[0])]]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[:,:,:,:,:,[l,(l+factors.shape[0])]]
            weighted_floor_warped_factors.append(weighted_floor_warped_factor_l)
            weighted_ceil_warped_factors.append(weighted_ceil_warped_factor_l)
        weighted_floor_warped_factors = torch.stack(weighted_floor_warped_factors)
        weighted_ceil_warped_factors = torch.stack(weighted_ceil_warped_factors)
        warped_factors = weighted_floor_warped_factors + weighted_ceil_warped_factors

        early = factors[:, :self.left_landmark1]
        early = early.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*early.shape,*warped_factors.shape[2:-1])
        mid = factors[:, self.right_landmark1:self.left_landmark2]
        mid = mid.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*mid.shape,*warped_factors.shape[2:-1])
        late = factors[:, self.right_landmark2:]
        late = late.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*late.shape,*warped_factors.shape[2:-1])
        warped_factors = torch.cat([early, warped_factors[:, :, :, :, :, :, 0], mid, warped_factors[:, :, :, :, :, :, 1], late], dim=1)

        return warped_factors


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors, n_areas):  # and first 2 entropy terms
        # Weight Matrices

        # warped_factors # L x T x M x N x R x C
        # Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        Y_times_N_matrix = torch.einsum('ktrc,ltmnrc->kltmnrc', Y, warped_factors)
        sum_Y_times_N_matrix = torch.sum(Y_times_N_matrix, dim=(2, 5))
        exp_N_matrix = torch.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x L x C
        # logterm2  # L x M x N x C
        sum_Y_term = torch.sum(Y, dim=(1, 2))  # K x C
        logterm1 = sum_Y_term[:, None, :] + F.softplus(self.alpha)[None, :, None]
        logterm2 = self.dt * torch.sum(exp_N_matrix, dim=(1, 4)) + F.softplus(self.theta)[:, None, None, None]
        # logterm # K x L x M x N x C
        logterm = torch.einsum('klc,lmnc->klmnc', logterm1, torch.log(logterm2))
        # alphalogtheta # 1 x L x 1 x 1 x 1
        alphalogtheta = (F.softplus(self.alpha) * torch.log(F.softplus(self.theta))).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # sum_Y_times_logalpha  # K x C x L
        sum_Y_times_logalpha = torch.einsum('kc,l->klc', sum_Y_term, torch.log(F.softplus(self.alpha)))
        # sum_Y_times_logalpha # K x L x 1 x 1 x C
        sum_Y_times_logalpha = sum_Y_times_logalpha[:, :, None, None, :]
        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = torch.log(F.softmax(torch.cat([torch.zeros(1), self.pi]), dim=0)).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alpha_expand = F.softplus(self.alpha).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        theta_expand = F.softplus(self.theta).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # U_tensor # K x L x M x N x C
        U_tensor = sum_Y_times_N_matrix - logterm + alphalogtheta + sum_Y_times_logalpha + logpi_expand
        # U_tensor # K x A x La x M x N x C
        L_a = U_tensor.shape[1] // n_areas
        U_tensor = U_tensor.reshape(U_tensor.shape[0], n_areas, L_a, U_tensor.shape[2], U_tensor.shape[3], U_tensor.shape[4])
        # W_CMNK_tensor # K x L x M x N x C
        W_CMNK_tensor = F.softmax(U_tensor, dim=2).reshape(sum_Y_times_N_matrix.shape)

        # logsumexp_U_tensor # K x A x M x N x C
        logsumexp_U_tensor = torch.logsumexp(U_tensor, dim=2)
        # neuron_area_access  #  K x A x 1 x 1 x C
        neuron_area_access = neuron_factor_access[:, [i * L_a for i in range(n_areas)], None, None, :]
        # sum_logsumexp_tensor #  M x N x C
        sum_logsumexp_U_tensor = torch.sum(neuron_area_access * logsumexp_U_tensor, dim=(0, 1))
        # W_C_tensor # 1 x 1 x M x N x C
        W_C_tensor = (F.softmax(sum_logsumexp_U_tensor.reshape(-1, sum_logsumexp_U_tensor.shape[-1]), dim=0).
                      reshape(sum_logsumexp_U_tensor.shape)).unsqueeze(0).unsqueeze(1)
        W_C_tensor = W_C_tensor.detach()
        self.W_C_tensor = W_C_tensor

        # neuron_factor_access  # K x L x C
        # W_tensor # K x L x M x N x C
        W_tensor = (neuron_factor_access.unsqueeze(2).unsqueeze(3) * W_CMNK_tensor * W_C_tensor).detach()

        # A_tensor # K x L x M x N x C
        A_tensor = torch.einsum('klc,lmnc->klmnc', logterm1, 1 / logterm2).detach()
        B_tensor = (torch.digamma(logterm1)[:, :, None, None, :] - torch.log(logterm2[None, :, :, :, :])).detach()

        # Liklelihood Terms
        elbo_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None, :, :, :, :] - torch.lgamma(alpha_expand) +
                     alpha_expand * (torch.log(theta_expand) + B_tensor) + logpi_expand)

        # elbo_term # K x L x M x N x C
        elbo_term = torch.sum(W_tensor * elbo_term)

        return elbo_term


    def compute_offset_entropy_terms(self): # last 2 entropy terms
        # Entropy1 Terms
        dim = self.config_peak_offset_stdevs.shape[0]

        Sigma1 = torch.diag(F.softplus(self.config_peak_offset_stdevs)) @ torch.diag(F.softplus(self.config_peak_offset_stdevs)).t()
        det_Sigma1 = torch.linalg.det(Sigma1)
        inv_Sigma1 = torch.linalg.inv(Sigma1)
        prod_term1 = torch.einsum('ncl,lj,ncj->nc', self.transformed_config_peak_offset_samples, inv_Sigma1, self.transformed_config_peak_offset_samples)  # sum over l
        # entropy_term1  # N x C
        entropy_term1 = -0.5 * (torch.log((2 * torch.pi) ** dim * det_Sigma1) + prod_term1)
        # entropy_term1  # 1 x 1 x 1 x N x C
        # W_C_tensor # 1 x 1 x M x N x C
        entropy_term1 = torch.sum(self.W_C_tensor * entropy_term1.unsqueeze(0).unsqueeze(1).unsqueeze(2))

        Sigma2 = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.t()
        det_Sigma2 = torch.linalg.det(Sigma2)
        inv_Sigma2 = torch.linalg.inv(Sigma2)
        prod_term2 = torch.einsum('mrcl,lj,mrcj->mrc', self.transformed_trial_peak_offset_samples, inv_Sigma2, self.transformed_trial_peak_offset_samples) # sum over l
        # entropy_term2  # M x C
        entropy_term2 = -0.5 * torch.sum(torch.log((2 * torch.pi)**dim * det_Sigma2) + prod_term2, dim=1) # sum over r
        # entropy_term2  # 1 x 1 x M x 1 x C
        # W_C_tensor # 1 x 1 x M x N x C
        entropy_term2 = torch.sum(self.W_C_tensor * entropy_term2.unsqueeze(0).unsqueeze(1).unsqueeze(3))
        entropy_term = entropy_term1 + entropy_term2
        return entropy_term


    def compute_penalty_terms(self, tau_beta, tau_budget, tau_sigma1, tau_sigma2):
        # Penalty Terms
        sigma_Penalty1 = - tau_sigma1 * self.config_peak_offset_stdevs @ self.config_peak_offset_stdevs

        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty2 = -tau_sigma2 * (torch.sum(torch.abs(inv_Sigma)) - torch.sum(torch.abs(torch.diag(inv_Sigma))))

        latent_factors = F.softplus(self.beta)
        smoothness_budget_constrained = F.softmax(torch.cat([torch.zeros(1), self.smoothness_budget]), dim=0)
        beta_s2_penalty = - tau_beta * smoothness_budget_constrained.t() @ torch.sum((latent_factors @ self.Delta2TDelta2) * latent_factors, dim=1)

        smoothness_budget_penalty = - tau_budget * (self.smoothness_budget @ self.smoothness_budget)

        penalty_term = sigma_Penalty1 + sigma_Penalty2 + beta_s2_penalty + smoothness_budget_penalty
        return penalty_term


    # def forward(self, Y, neuron_factor_access, tau_beta, tau_budget, tau_sigma1, tau_sigma2):
    #     _, _, n_trials, n_configs = Y.shape
    #     warped_factors = self.warp_all_latent_factors_for_all_trials(n_configs, n_trials)
    #     likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
    #     entropy_term = self.compute_offset_entropy_terms()
    #     penalty_term = self.compute_penalty_terms(tau_beta, tau_budget, tau_sigma1, tau_sigma2)
    #     return likelihood_term, entropy_term, penalty_term, warped_factors
