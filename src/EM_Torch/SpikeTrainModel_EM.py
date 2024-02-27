import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline
from src.psplines_gradient_method.general_functions import create_second_diff_matrix
import numpy as np


class LikelihoodModel(nn.Module):
    def __init__(self):
        super(LikelihoodModel, self).__init__()

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
        self.transformed_trial_peak_offset_samples = None  # CR x 2AL
        self.transformed_config_peak_offset_samples = None  # C x 2AL
        self.n_config_samples = None
        self.n_trial_samples = None
        self.W_C_tensor = None  # 1 x 1 x M x N x C
        self.BDelta2TDelta2BT = None  # T x T # tikhonov regularization

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offset_stdevs = None  # 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.smoothness_budget = None  # L x 1


    def initialize(self, Y, time, factor_access, n_trial_samples, n_config_samples):
        self.Y = torch.tensor(Y)
        self.time = torch.tensor(time)
        self.neuron_factor_access = torch.tensor(factor_access)
        dt = round(time[1] - time[0], 3)
        self.dt = torch.tensor(dt)
        K, T, n_trials, n_configs = Y.shape
        C, K, n_factors = factor_access.shape

        # fixed values
        self.n_config_samples = n_config_samples
        self.n_trial_samples = n_trial_samples
        degree = 3
        knots = np.concatenate([np.repeat(time[0], degree), time, np.repeat(time[-1], degree)])
        knots[-1] = knots[-1] + dt
        Bsplines = BSpline.design_matrix(time, knots, degree).transpose().toarray()
        Delta2BT = create_second_diff_matrix(T) @ Bsplines.T
        self.BDelta2TDelta2BT = torch.tensor(Delta2BT.T @ Delta2BT).to_sparse()

        # Paremeters
        self.beta = nn.Parameter(torch.randn(n_factors, self.time.shape[0]))
        self.alpha = nn.Parameter(torch.randn(n_factors))
        self.theta = nn.Parameter(torch.randn(n_factors))
        self.pi = nn.Parameter(torch.randn(n_factors-1))
        self.config_peak_offset_stdevs = nn.Parameter(torch.randn(2 * n_factors))
        bounds = 0.05
        matrix = torch.tril(torch.randn(2 * n_factors, 2 * n_factors))
        # Ensure diagonal elements are positive
        for i in range(min(matrix.size())):
            matrix[i, i] += F.softplus(matrix[i, i])
        # Make it a learnable parameter
        self.trial_peak_offset_covar_ltri = nn.Parameter(matrix)
        self.smoothness_budget = nn.Parameter(torch.zeros(n_factors-1))
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (np.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())
        # std_dev = np.sqrt(np.diag(trial_peak_offset_covar_matrix))
        # corr = np.diag(1/std_dev) @ trial_peak_offset_covar_matrix @ np.diag(1/std_dev)
        return self


    def init_ground_truth(self, latent_factors, alpha, theta, pi, config_peak_offset_stdevs, trial_peak_offset_covar_ltri):
        self.beta = nn.Parameter(latent_factors)
        self.alpha = nn.Parameter(alpha)
        self.theta = nn.Parameter(theta)
        self.pi = nn.Parameter(pi)
        self.config_peak_offset_stdevs = nn.Parameter(config_peak_offset_stdevs)
        self.trial_peak_offset_covar_ltri = nn.Parameter(trial_peak_offset_covar_ltri)


    def warp_all_latent_factors_for_all_trials(self, n_configs):
        K, T, n_trials, C = self.Y.shape
        n_factors = self.beta.shape[0]
        config_peak_offset_samples = torch.randn(self.n_config_samples, n_configs, 2 * n_factors)
        trial_peak_offset_samples = torch.randn(self.n_trial_samples, n_trials * n_configs, 2 * n_factors).view(
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


    def compute_log_elbo(self, config_indcs): # and first 2 entropy terms
        # Weight Matrices
        n_configs = len(config_indcs)
        warped_factors = self.warp_all_latent_factors_for_all_trials(n_configs)
        # warped_factors # L x T x M x N x R x C
        # self.Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        # neuron_factor_access  #  C x K x L
        Y = self.Y[:,:,:,config_indcs]
        neuron_factor_access = self.neuron_factor_access[config_indcs,:,:]
        Y_times_N_matrix = torch.einsum('ktrc,ckl,ltmnrc->kltmnrc', Y, neuron_factor_access, warped_factors)
        sum_Y_times_N_matrix = torch.sum(Y_times_N_matrix, dim=(2, 5))
        exp_N_matrix = torch.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x C x L
        # logterm2  # L x M x N x C
        sum_Y_term = torch.sum(Y, dim=(1,2)) # K x C
        logterm1 = sum_Y_term[:,:,None] + F.softplus(self.alpha)[None,None,:]
        logterm2 = self.dt * torch.sum(exp_N_matrix, dim=(1, 4)) + F.softplus(self.theta)[:,None,None,None]
        # logterm # K x L x M x N x C
        logterm = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, torch.log(logterm2))
        # alphalogtheta # 1 x L x 1 x 1 x 1
        alphalogtheta = (F.softplus(self.alpha) * torch.log(F.softplus(self.theta))).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # sum_Y_times_logalpha  # K x C x L
        sum_Y_times_logalpha = torch.einsum('kc,ckl,l->klc', sum_Y_term, neuron_factor_access, torch.log(F.softplus(self.alpha)))
        # sum_Y_times_logalpha # K x L x 1 x 1 x C
        sum_Y_times_logalpha = sum_Y_times_logalpha[:,:,None,None,:]
        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = torch.log(F.softmax(torch.cat([torch.zeros(1), self.pi]), dim=0)).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alpha_expand = F.softplus(self.alpha).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        theta_expand = F.softplus(self.theta).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # U_tensor # K x L x M x N x C
        U_tensor = torch.exp(sum_Y_times_N_matrix - logterm + alphalogtheta + sum_Y_times_logalpha + logpi_expand)

        # W_CMNK_tensor # K x L x M x N x C
        W_CMNK_tensor = F.softmax(U_tensor, dim=1)

        exp_sum_logsumexp_tensor = torch.sum(torch.logsumexp(U_tensor, dim=1), dim=0)
        exp_sum_logsumexp_tensor_reshape = exp_sum_logsumexp_tensor.reshape(-1, W_CMNK_tensor.shape[-1])

        # W_C_tensor # 1 x 1 x M x N x C
        W_C_tensor = F.softmax(exp_sum_logsumexp_tensor_reshape, dim=0).reshape(exp_sum_logsumexp_tensor.shape)[None,None,:,:,:]
        self.W_C_tensor = W_C_tensor.detach()

        # W_tensor # K x L x M x N x C
        W_tensor = (W_CMNK_tensor * W_C_tensor).detach()

        # A_tensor # K x L x M x N x C
        A_tensor = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, 1/logterm2)

        # Liklelihood Terms
        elbo_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None,:,:,:,:] - torch.lgamma(alpha_expand) + alpha_expand *
                           (torch.log(theta_expand) + torch.permute(torch.digamma(logterm1), (0, 2, 1))[:,:,None,None,:] -
                            torch.log(logterm2[None,:,:,:,:])) + logpi_expand)
        elbo_term = torch.sum(W_tensor * elbo_term)

        return elbo_term


    def compute_offset_entropy_terms(self): # last 2 entropy terms
        # Entropy1 Terms
        dim = self.config_peak_offset_stdevs.shape[0]
        prod_sq = torch.einsum('ncl,l->ncl', self.transformed_config_peak_offset_samples, 1/F.softplus(self.config_peak_offset_stdevs))**2
        # entropy_term1  # N x C
        entropy_term1 = -0.5 * torch.sum(torch.log((2 * torch.pi)**dim * torch.prod(F.softplus(self.config_peak_offset_stdevs)**2)) + prod_sq, dim=2) # sum over l
        # entropy_term1  # 1 x 1 x 1 x N x C
        # W_C_tensor # 1 x 1 x M x N x C
        entropy_term1 = torch.sum(self.W_C_tensor * entropy_term1.unsqueeze(0).unsqueeze(1).unsqueeze(2))

        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        det_Sigma = torch.linalg.det(Sigma)
        inv_Sigma = torch.linalg.inv(Sigma)
        prod_term = torch.einsum('mrcl,lj,mrcj->mrc', self.transformed_trial_peak_offset_samples, inv_Sigma, self.transformed_trial_peak_offset_samples) # sum over l
        # entropy_term2  # M x C
        entropy_term2 = -0.5 * torch.sum(torch.log((2 * torch.pi)**dim * det_Sigma) + prod_term, dim=1) # sum over r
        # entropy_term2  # 1 x 1 x M x 1 x C
        # W_C_tensor # 1 x 1 x M x N x C
        entropy_term2 = torch.sum(self.W_C_tensor * entropy_term2.unsqueeze(0).unsqueeze(1).unsqueeze(3))
        entropy_term = entropy_term1 + entropy_term2
        return entropy_term


    def compute_penalty_terms(self, tau_beta, tau_budget, tau_sigma):
        # Penalty Terms
        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty = - tau_sigma * torch.sum(torch.abs(inv_Sigma))
        latent_factors = torch.cat([torch.tensor([0]).unsqueeze(0).expand(self.beta.shape[0],-1),
                                    F.softplus(self.beta),
                                    torch.tensor([0]).unsqueeze(0).expand(self.beta.shape[0],-1)],
                                   dim=1).float()
        smoothness_budget_constrained = F.softmax(torch.cat([torch.zeros(1), self.smoothness_budget]), dim=0)
        beta_s2_penalty = - tau_beta * smoothness_budget_constrained.float().t() @ torch.sum((latent_factors @ self.BDelta2TDelta2BT.float()) * latent_factors, dim=1)

        smoothness_budget_penalty = - tau_budget * (self.smoothness_budget.t() @ self.smoothness_budget)

        penalty_term = sigma_Penalty + beta_s2_penalty + smoothness_budget_penalty
        return penalty_term


    def forward(self, config_indcs, tau_beta, tau_budget, tau_sigma):
        likelihood_term = self.compute_log_elbo(config_indcs)
        entropy_term = self.compute_offset_entropy_terms()
        penalty_term = self.compute_penalty_terms(tau_beta, tau_budget, tau_sigma)
        return likelihood_term, entropy_term, penalty_term
