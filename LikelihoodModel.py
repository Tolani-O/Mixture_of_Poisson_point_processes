import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix, inv_softplus
import numpy as np
from scipy.ndimage import gaussian_filter1d


class LikelihoodModel(nn.Module):
    def __init__(self, time):
        super(LikelihoodModel, self).__init__()

        self.time = torch.tensor(time)
        dt = round(time[1] - time[0], 3)
        self.dt = torch.tensor(dt)
        landmark_spread = 50
        self.left_landmark1 = 20
        self.mid_landmark1 = self.left_landmark1 + landmark_spread / 2
        self.right_landmark1 = self.left_landmark1 + landmark_spread
        self.left_landmark2 = 120
        self.mid_landmark2 = self.left_landmark2 + landmark_spread / 2
        self.right_landmark2 = self.left_landmark2 + landmark_spread
        self.neuron_factor_access = None
        self.transformed_trial_peak_offset_samples = None  # CR x 2AL
        self.transformed_config_peak_offset_samples = None  # C x 2AL
        self.warped_factors = None  # L x T x M x N x R x C
        self.neuron_factor_assignment = None  # C x K x L

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offset_stdevs = None  # 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL


    def init_params(self, beta, alpha, theta, pi, config_peak_offset_stdevs, trial_peak_offset_covar_ltri,
                    n_configs, n_neurons):
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.pi = pi
        self.config_peak_offset_stdevs = config_peak_offset_stdevs
        self.trial_peak_offset_covar_ltri = trial_peak_offset_covar_ltri
        n_factors = self.beta.shape[0]
        self.neuron_factor_assignment = torch.zeros((n_configs, n_neurons, n_factors))


    def init_random(self, n_configs, n_trials):
        n_factors = self.beta.shape[0]
        config_peak_offset_samples = torch.randn(1, n_configs, 2*n_factors)
        trial_peak_offset_samples = torch.randn(1, n_trials*n_configs, 2*n_factors).view(1, n_trials, n_configs, 2*n_factors)
        transformed_trial_peak_offset_samples = torch.einsum('lj,mrcj->mrcl', self.trial_peak_offset_covar_ltri,
                                                             trial_peak_offset_samples)
        transformed_config_peak_offset_samples = torch.einsum('l,ncl->ncl', F.softplus(self.config_peak_offset_stdevs),
                                                              config_peak_offset_samples)
        self.transformed_trial_peak_offset_samples = nn.Parameter(transformed_trial_peak_offset_samples)
        self.transformed_config_peak_offset_samples = nn.Parameter(transformed_config_peak_offset_samples)


    def init_ground_truth(self, config_peak_offsets, trial_peak_offsets):
        self.transformed_config_peak_offset_samples = nn.Parameter(config_peak_offsets)
        self.transformed_trial_peak_offset_samples = nn.Parameter(trial_peak_offsets)


    def warp_all_latent_factors_for_all_trials(self):
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
        left_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.left_landmark1, self.left_landmark2]),
                                                            s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        right_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.right_landmark1, self.right_landmark2]),
                                                             s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
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
            floor_warped_factor_l = factors[l, floor_warped_indices[:, :, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[:, :, :, :, :, [l, (l + factors.shape[0])]]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[:, :, :, :, :, [l, (l + factors.shape[0])]]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[:, :, :, :, :, [l, (l + factors.shape[0])]]
            weighted_floor_warped_factors.append(weighted_floor_warped_factor_l)
            weighted_ceil_warped_factors.append(weighted_ceil_warped_factor_l)
        weighted_floor_warped_factors = torch.stack(weighted_floor_warped_factors)
        weighted_ceil_warped_factors = torch.stack(weighted_ceil_warped_factors)
        warped_factors = weighted_floor_warped_factors + weighted_ceil_warped_factors

        early = factors[:, :self.left_landmark1]
        early = early.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*early.shape, *warped_factors.shape[2:-1])
        mid = factors[:, self.right_landmark1:self.left_landmark2]
        mid = mid.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*mid.shape, *warped_factors.shape[2:-1])
        late = factors[:, self.right_landmark2:]
        late = late.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(*late.shape, *warped_factors.shape[2:-1])
        warped_factors = torch.cat([early, warped_factors[:, :, :, :, :, :, 0], mid, warped_factors[:, :, :, :, :, :, 1], late], dim=1)

        return warped_factors


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors):  # and first 2 entropy terms
        # Weight Matrices

        # warped_factors # L x T x M x N x R x C
        # Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        Y_times_N_matrix = torch.einsum('ktrc,ckl,ltmnrc->kltmnrc', Y, neuron_factor_access, warped_factors)
        sum_Y_times_N_matrix = torch.sum(Y_times_N_matrix, dim=(2, 5))
        exp_N_matrix = torch.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x C x L
        # logterm2  # L x M x N x C
        sum_Y_term = torch.sum(Y, dim=(1, 2))  # K x C
        logterm1 = sum_Y_term[:, :, None] + F.softplus(self.alpha)[None, None, :]
        logterm2 = self.dt * torch.sum(exp_N_matrix, dim=(1, 4)) + F.softplus(self.theta)[:, None, None, None]
        # logterm # K x L x M x N x C
        logterm = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, torch.log(logterm2))
        # alphalogtheta # 1 x L x 1 x 1 x 1
        alphalogtheta = (F.softplus(self.alpha) * torch.log(F.softplus(self.theta))).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # sum_Y_times_logalpha  # K x C x L
        sum_Y_times_logalpha = torch.einsum('kc,ckl,l->klc', sum_Y_term, neuron_factor_access, torch.log(F.softplus(self.alpha)))
        # sum_Y_times_logalpha # K x L x 1 x 1 x C
        sum_Y_times_logalpha = sum_Y_times_logalpha[:, :, None, None, :]
        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = torch.log(F.softmax(torch.cat([torch.zeros(1), self.pi]), dim=0)).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alpha_expand = F.softplus(self.alpha).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        theta_expand = F.softplus(self.theta).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # U_tensor # K x L x M x N x C
        U_tensor = torch.exp(sum_Y_times_N_matrix - logterm + alphalogtheta + sum_Y_times_logalpha + logpi_expand)

        # W_CMNK_tensor # K x L x M x N x C
        W_CMNK_tensor = F.softmax(U_tensor, dim=1)

        # W_tensor # K x L x M x N x C
        # neuron_factor_access  #  C x K x L
        W_tensor = (torch.permute(neuron_factor_access, (1, 2, 0)).unsqueeze(2).unsqueeze(3) * W_CMNK_tensor).detach()

        # A_tensor # K x L x M x N x C
        A_tensor = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, 1 / logterm2).detach()
        B_tensor = (torch.permute(torch.digamma(logterm1), (0, 2, 1))[:, :, None, None, :] -
                    torch.log(logterm2[None, :, :, :, :])).detach()

        # Liklelihood Terms
        elbo_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None, :, :, :, :] - torch.lgamma(alpha_expand) +
                     alpha_expand * (torch.log(theta_expand) + B_tensor) + logpi_expand)

        # elbo_term # K x L x M x N x C
        elbo_term = torch.sum(W_tensor * elbo_term)

        return elbo_term


    def compute_log_elbo_peak_times(self, Y, neuron_factor_access, warped_factors):  # and first 2 entropy terms
        # Weight Matrices

        # warped_factors # L x T x M x N x R x C
        # Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        Y_times_N_matrix = torch.einsum('ktrc,ckl,ltmnrc->kltmnrc', Y, neuron_factor_access, warped_factors)
        sum_Y_times_N_matrix = torch.sum(Y_times_N_matrix, dim=(2, 5))
        exp_N_matrix = torch.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x C x L
        # logterm2  # L x M x N x C
        sum_Y_term = torch.sum(Y, dim=(1, 2))  # K x C
        logterm1 = sum_Y_term[:, :, None] + F.softplus(self.alpha)[None, None, :]
        logterm2 = self.dt * torch.sum(exp_N_matrix, dim=(1, 4)) + F.softplus(self.theta)[:, None, None, None]
        # logterm # K x L x M x N x C
        logterm = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, torch.log(logterm2))
        # alphalogtheta # 1 x L x 1 x 1 x 1
        alphalogtheta = (F.softplus(self.alpha) * torch.log(F.softplus(self.theta))).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # sum_Y_times_logalpha  # K x C x L
        sum_Y_times_logalpha = torch.einsum('kc,ckl,l->klc', sum_Y_term, neuron_factor_access, torch.log(F.softplus(self.alpha)))
        # sum_Y_times_logalpha # K x L x 1 x 1 x C
        sum_Y_times_logalpha = sum_Y_times_logalpha[:, :, None, None, :]
        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = torch.log(F.softmax(torch.cat([torch.zeros(1), self.pi]), dim=0)).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # U_tensor # K x L x M x N x C
        U_tensor = torch.exp(sum_Y_times_N_matrix - logterm + alphalogtheta + sum_Y_times_logalpha + logpi_expand)

        # W_CMNK_tensor # K x L x M x N x C
        W_CMNK_tensor = F.softmax(U_tensor, dim=1)

        # W_tensor # K x L x M x N x C
        # neuron_factor_access  #  C x K x L
        W_tensor = (torch.permute(neuron_factor_access, (1, 2, 0)).unsqueeze(2).unsqueeze(3) * W_CMNK_tensor).detach()

        # A_tensor # K x L x M x N x C
        A_tensor = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, 1 / logterm2).detach()

        # Liklelihood Terms
        elbo_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None, :, :, :, :])

        # elbo_term # K x L x M x N x C
        elbo_term = torch.sum(W_tensor * elbo_term)

        return elbo_term


    def compute_log_elbo_factor_assignments(self, Y, neuron_factor_access, warped_factors):  # and first 2 entropy terms
        # Weight Matrices

        # warped_factors # L x T x M x N x R x C
        # Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        Y_times_N_matrix = torch.einsum('ktrc,ckl,ltmnrc->kltmnrc', Y, neuron_factor_access, warped_factors)
        sum_Y_times_N_matrix = torch.sum(Y_times_N_matrix, dim=(2, 5))
        exp_N_matrix = torch.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x C x L
        # logterm2  # L x M x N x C
        sum_Y_term = torch.sum(Y, dim=(1, 2))  # K x C
        logterm1 = sum_Y_term[:, :, None] + F.softplus(self.alpha)[None, None, :]
        logterm2 = self.dt * torch.sum(exp_N_matrix, dim=(1, 4)) + F.softplus(self.theta)[:, None, None, None]

        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = torch.log(F.softmax(torch.cat([torch.zeros(1), self.pi]), dim=0)).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alpha_expand = F.softplus(self.alpha).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        theta_expand = F.softplus(self.theta).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        W_tensor = torch.permute(neuron_factor_access, (1, 2, 0)).unsqueeze(2).unsqueeze(3)
        # A_tensor # K x L x M x N x C
        A_tensor = torch.einsum('kcl,ckl,lmnc->klmnc', logterm1, neuron_factor_access, 1 / logterm2).detach()
        B_tensor = (torch.permute(torch.digamma(logterm1), (0, 2, 1))[:, :, None, None, :] -
                    torch.log(logterm2[None, :, :, :, :])).detach()

        # Liklelihood Terms
        elbo_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None, :, :, :, :] - torch.lgamma(alpha_expand) +
                     alpha_expand * (torch.log(theta_expand) + B_tensor) + logpi_expand)

        # elbo_term # K x L x M x N x C
        elbo_term = W_tensor * elbo_term

        return elbo_term


    def compute_offset_entropy_terms(self):  # last 2 entropy terms
        # Entropy1 Terms
        dim = self.config_peak_offset_stdevs.shape[0]

        Sigma1 = torch.diag(F.softplus(self.config_peak_offset_stdevs)) @ torch.diag(F.softplus(self.config_peak_offset_stdevs)).t()
        det_Sigma1 = torch.linalg.det(Sigma1)
        inv_Sigma1 = torch.linalg.inv(Sigma1)
        prod_term1 = torch.einsum('ncl,lj,ncj->nc', self.transformed_config_peak_offset_samples, inv_Sigma1, self.transformed_config_peak_offset_samples)  # sum over l
        # entropy_term1  # N x C
        entropy_term1 = -0.5 * torch.sum(torch.log((2 * torch.pi) ** dim * det_Sigma1) + prod_term1)

        Sigma2 = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.t()
        det_Sigma2 = torch.linalg.det(Sigma2)
        inv_Sigma2 = torch.linalg.inv(Sigma2)
        prod_term2 = torch.einsum('mrcl,lj,mrcj->mrc', self.transformed_trial_peak_offset_samples, inv_Sigma2, self.transformed_trial_peak_offset_samples)  # sum over l
        # entropy_term2  # M x C
        entropy_term2 = -0.5 * torch.sum(torch.log((2 * torch.pi) ** dim * det_Sigma2) + prod_term2)

        entropy_term = entropy_term1 + entropy_term2
        return entropy_term


    def compute_neuron_factor_assignment(self, Y, neuron_factor_access, warped_factors):
        likelihoods = self.compute_log_elbo_factor_assignments(Y, neuron_factor_access, warped_factors)
        # likelihoods # K x L x M x N x C
        likelihoods = torch.permute(likelihoods.squeeze(), (2, 0, 1))
        # likelihoods # C x K x L
        self.neuron_factor_assignment = torch.where(likelihoods == torch.max(likelihoods, dim=2, keepdim=True).values, 1, 0)
        return self.neuron_factor_assignment


    def forward(self, Y, neuron_factor_access):
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
        entropy_term = self.compute_offset_entropy_terms()
        return likelihood_term, entropy_term
