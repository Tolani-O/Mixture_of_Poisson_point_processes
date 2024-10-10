import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix
import numpy as np
import pandas as pd
import pickle


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trials, n_trial_samples,
                 peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks):
        super(LikelihoodELBOModel, self).__init__()

        self.device = 'cpu'
        self.time = torch.tensor(time)
        dt = round(time[1] - time[0], 3)
        self.dt = torch.tensor(dt)
        T = time.shape[0]
        self.peak1_left_landmarks = torch.searchsorted(self.time, torch.tensor(peak1_left_landmarks), side='left')-1
        self.peak1_left_landmarks = torch.cat([self.peak1_left_landmarks] * n_areas)
        self.peak1_right_landmarks = torch.searchsorted(self.time, torch.tensor(peak1_right_landmarks), side='left')
        self.peak1_right_landmarks = torch.cat([self.peak1_right_landmarks] * n_areas)
        self.peak2_left_landmarks = torch.searchsorted(self.time, torch.tensor(peak2_left_landmarks), side='left')-1
        self.peak2_left_landmarks = torch.cat([self.peak2_left_landmarks] * n_areas)
        self.peak2_right_landmarks = torch.searchsorted(self.time, torch.tensor(peak2_right_landmarks), side='left')
        self.peak2_right_landmarks = torch.cat([self.peak2_right_landmarks] * n_areas)
        self.n_factors = n_factors
        self.n_areas = n_areas
        self.n_trial_samples = n_trial_samples
        self.n_configs = n_configs
        self.n_trials = n_trials
        Delta2 = create_second_diff_matrix(T)
        self.Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2)  # T x T # tikhonov regularization

        # Storage for use in the forward pass
        self.W_CKL = None  # C x K x L
        self.trial_peak_offset_proposal_samples = None  # N x R x C x 2AL
        self.a_CKL = None  # C x K x L
        self.theta_init = None  # 1 x AL
        self.pi_init = None  # 1 x AL

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri_diag = None
        self.trial_peak_offset_covar_ltri_offdiag = None
        self.trial_peak_offset_proposal_means = None  # R x C x 2AL
        self.trial_peak_offset_proposal_sds = None  # 2AL


    def init_random(self):
        self.beta = nn.Parameter(torch.log(torch.randn(self.n_factors, self.time.shape[0]-1, dtype=torch.float64)))
        self.alpha = nn.Parameter(torch.randn(self.n_factors, dtype=torch.float64))
        self.config_peak_offsets = nn.Parameter(torch.randn(self.n_configs, 2 * self.n_factors, dtype=torch.float64))
        n_dims = 2 * self.n_factors
        num_elements = n_dims * (n_dims - 1) // 2
        self.trial_peak_offset_covar_ltri_diag = nn.Parameter(torch.rand(n_dims, dtype=torch.float64)+1)
        self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(torch.randn(num_elements, dtype=torch.float64))
        self.trial_peak_offset_proposal_means = nn.Parameter(torch.randn(self.n_trials, self.n_configs, 2 * self.n_factors, dtype=torch.float64))
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.rand(n_dims, dtype=torch.float64) + 1)
        self.standard_init()


    def init_zero(self):
        self.beta = nn.Parameter(torch.zeros(self.n_factors, self.time.shape[0]-1, dtype=torch.float64))
        self.alpha = nn.Parameter(torch.ones(self.n_factors, dtype=torch.float64))
        self.config_peak_offsets = nn.Parameter(torch.zeros(self.n_configs, 2 * self.n_factors, dtype=torch.float64))
        n_dims = 2 * self.n_factors
        num_elements = n_dims * (n_dims - 1) // 2
        self.trial_peak_offset_covar_ltri_diag = nn.Parameter(torch.ones(n_dims, dtype=torch.float64))
        self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(torch.zeros(num_elements, dtype=torch.float64))
        self.trial_peak_offset_proposal_means = nn.Parameter(torch.zeros(self.n_trials, self.n_configs, n_dims, dtype=torch.float64))
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.ones(n_dims, dtype=torch.float64))
        self.standard_init()


    def standard_init(self):
        self.theta_init = torch.ones(self.n_factors, dtype=torch.float64)
        self.pi_init = F.softmax(torch.zeros(self.n_areas, self.n_factors // self.n_areas, dtype=torch.float64), dim=1).flatten()
        self.W_CKL = None
        self.a_CKL = None


    def init_ground_truth(self, beta=None, alpha=None, theta=None, pi=None,
                          trial_peak_offset_proposal_means=None, trial_peak_offset_proposal_sds=None,
                          config_peak_offsets=None, trial_peak_offset_covar_ltri=None,
                          W_CKL=None, init='zeros'):
        if init == 'zeros':
            self.init_zero()
        elif init == 'random':
            self.init_random()
        if beta is not None:
            self.beta = nn.Parameter(beta)
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
        if theta is not None:
            self.theta_init = theta
        if W_CKL is not None:
            self.W_CKL = W_CKL
        elif pi is not None:
            self.pi_init = pi
        if trial_peak_offset_proposal_means is not None:
            self.trial_peak_offset_proposal_means = nn.Parameter(trial_peak_offset_proposal_means)
        if trial_peak_offset_proposal_sds is not None:
            self.trial_peak_offset_proposal_sds = nn.Parameter(trial_peak_offset_proposal_sds)
        if config_peak_offsets is not None:
            self.config_peak_offsets = nn.Parameter(config_peak_offsets)
        if trial_peak_offset_covar_ltri is not None:
            self.trial_peak_offset_covar_ltri_diag = nn.Parameter(trial_peak_offset_covar_ltri.diag())
            n_dims = 2 * self.n_factors
            indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
            self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(trial_peak_offset_covar_ltri[indices[0], indices[1]])


    def init_from_data(self, Y, factor_access, sd_init, cluster_dir, init='zeros'):
        Y, factor_access = Y.cpu(), factor_access.cpu()
        # Y # K x T x R x C
        # factor_access  # K x L x C
        _, _, R, _ = Y.shape
        cluster_dir = os.path.join(cluster_dir, 'cluster_initialization.pkl')
        if not os.path.exists(cluster_dir):
            raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
        print('Loading clusters from: ', cluster_dir)
        with open(cluster_dir, 'rb') as f:
            data = pickle.load(f)
        neuron_factor_assignment, beta = data['neuron_factor_assignment'], data['beta']
        spike_counts = torch.einsum('ktrc,klc->krlc', Y, factor_access)
        avg_spike_counts = torch.sum(spike_counts, dim=(0,1,3)) / (R * torch.sum(factor_access, dim=(0, 2)))
        print('Average spike counts:')
        print(avg_spike_counts.reshape(self.n_areas, -1).cpu().numpy())
        centered_spike_counts = torch.einsum('krlc,klc->krlc', spike_counts - avg_spike_counts.unsqueeze(0).unsqueeze(1).unsqueeze(3), factor_access)
        spike_ct_var = torch.sum(centered_spike_counts**2, dim=(0,1,3)) / ((R * torch.sum(factor_access, dim=(0, 2)))-1)
        print('Spike count variance - Average spike counts:')
        print((spike_ct_var-avg_spike_counts).reshape(self.n_areas, -1).cpu().numpy())
        alpha = (avg_spike_counts)**2/(spike_ct_var-avg_spike_counts)
        alpha = alpha.expm1().clamp_min(1e-6).log()
        theta = avg_spike_counts/(spike_ct_var-avg_spike_counts)
        trial_peak_offset_proposal_sds = sd_init * torch.ones(2*self.n_factors, dtype=torch.float64)
        self.init_ground_truth(beta=beta, alpha=alpha, theta=theta,
                               trial_peak_offset_proposal_sds=trial_peak_offset_proposal_sds,
                               W_CKL=neuron_factor_assignment, init=init)


    def cuda(self, device=None):
        self.device = 'cuda'
        self.time = self.time.cuda(device)
        self.theta_init = self.theta_init.cuda(device)
        self.pi_init = self.pi_init.cuda(device)
        self.Delta2TDelta2 = self.Delta2TDelta2.cuda(device)
        self.peak1_left_landmarks = self.peak1_left_landmarks.cuda(device)
        self.peak2_left_landmarks = self.peak2_left_landmarks.cuda(device)
        self.peak1_right_landmarks = self.peak1_right_landmarks.cuda(device)
        self.peak2_right_landmarks = self.peak2_right_landmarks.cuda(device)
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cuda(device)
        if self.a_CKL is not None:
            self.a_CKL = self.a_CKL.cuda(device)
        super(LikelihoodELBOModel, self).cuda(device)
        return self


    def cpu(self):
        self.device = 'cpu'
        self.time = self.time.cpu()
        self.theta_init = self.theta_init.cpu()
        self.pi_init = self.pi_init.cpu()
        self.Delta2TDelta2 = self.Delta2TDelta2.cpu()
        self.peak1_left_landmarks = self.peak1_left_landmarks.cpu()
        self.peak2_left_landmarks = self.peak2_left_landmarks.cpu()
        self.peak1_right_landmarks = self.peak1_right_landmarks.cpu()
        self.peak2_right_landmarks = self.peak2_right_landmarks.cpu()
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cpu()
        if self.a_CKL is not None:
            self.a_CKL = self.a_CKL.cpu()
        super(LikelihoodELBOModel, self).cpu()
        return self


    def ltri_matix(self, device=None):
        if device is None:
            device = self.device
        n_dims = 2 * self.n_factors
        ltri_matrix = torch.zeros(n_dims, n_dims, dtype=torch.float64, device=device)
        ltri_matrix[torch.arange(n_dims), torch.arange(n_dims)] = self.trial_peak_offset_covar_ltri_diag
        indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
        ltri_matrix[indices[0], indices[1]] = self.trial_peak_offset_covar_ltri_offdiag
        return ltri_matrix

    def sd_matrix(self, device=None):
        if device is None:
            device = self.device
        n_dims = 2 * self.n_factors
        sd_matrix = torch.zeros(n_dims, n_dims, dtype=torch.float64, device=device)
        sd_matrix[torch.arange(n_dims), torch.arange(n_dims)] = self.trial_peak_offset_proposal_sds
        return sd_matrix

    def unnormalized_log_factors(self):
        return torch.cat([torch.zeros(self.n_factors, 1, device=self.device, dtype=torch.float64), self.beta], dim=1)

    def theta_value(self):
        if self.W_CKL is None or self.a_CKL is None:
            return self.theta_init
        # W_L  # L
        W_L = torch.sum(self.W_CKL, dim=(0, 1))
        # Wa_L  # L
        Wa_L = torch.sum(self.W_CKL * self.a_CKL, dim=(0, 1))
        theta = F.softplus(self.alpha) * (W_L / Wa_L)
        return theta.detach()

    def validated_W_CKL(self, W_CKL, neuron_factor_access):
        W_L = torch.sum(W_CKL, dim=(0, 1)).reshape(self.n_areas, -1)
        neuron_factor_access_L = torch.sum(neuron_factor_access, dim=(0, 1)).reshape(self.n_areas, -1)
        W_CKL_L = W_CKL.reshape(W_CKL.shape[0], W_CKL.shape[1], self.n_areas, -1) + 0
        for i in range(W_L.shape[0]):
            while torch.any(W_L[i] < 1):
                min_idx = torch.argmin(W_L[i])
                max_idx = torch.argmax(W_L[i])
                W_L[i, min_idx] += 1
                W_L[i, max_idx] -= 1
                W_CKL_L[:, :, i, min_idx] += 1/neuron_factor_access_L[i, min_idx]
                W_CKL_L[:, :, i, max_idx] -= 1/neuron_factor_access_L[i, max_idx]
        W_CKL = W_CKL_L.reshape(*W_CKL.shape[:2], -1) * neuron_factor_access
        return W_CKL

    def pi_value(self, neuron_factor_access):
        if self.W_CKL is None:
            return self.pi_init
        # W_L  # L
        W_L = torch.sum(self.W_CKL, dim=(0, 1))
        pi = W_L / torch.sum(neuron_factor_access, dim=(0, 1))
        return pi.detach()

    def generate_trial_peak_offset_samples(self):
        gaussian_sample = torch.randn(self.n_trial_samples, self.n_trials, self.n_configs, 2 * self.n_factors,
                                      device=self.device, dtype=torch.float64)
        sd_matrix = self.sd_matrix()
        # trial_peak_offset_proposal_samples N x R x C x 2AL
        self.trial_peak_offset_proposal_samples = (self.trial_peak_offset_proposal_means.unsqueeze(0) +
                                                   torch.einsum('lj,nrcj->nrcl', sd_matrix, gaussian_sample)).detach()


    def generate_trial_peak_offset_single_sample(self):
        self.trial_peak_offset_proposal_samples = self.trial_peak_offset_proposal_means.unsqueeze(0).detach()


    def warp_all_latent_factors_for_all_trials(self):
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks()
        warped_times = self.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors = self.compute_warped_factors(warped_times)
        return warped_factors


    def compute_offsets_and_landmarks(self):
        factors = torch.exp(self.unnormalized_log_factors())
        avg_peak1_times = self.time[torch.tensor([self.peak1_left_landmarks[i] + torch.argmax(factors[i, self.peak1_left_landmarks[i]:self.peak1_right_landmarks[i]])
                                                  for i in range(self.peak1_left_landmarks.shape[0])])]
        avg_peak2_times = self.time[torch.tensor([self.peak2_left_landmarks[i] + torch.argmax(factors[i, self.peak2_left_landmarks[i]:self.peak2_right_landmarks[i]])
                                                  for i in range(self.peak2_left_landmarks.shape[0])])]
        avg_peak_times = torch.cat([avg_peak1_times, avg_peak2_times])
        # avg_peak_times  # 2AL
        # self.trial_peak_offset_proposal_samples # N x R x C x 2AL
        # self.config_peak_offsets  # C x 2AL
        offsets = self.trial_peak_offset_proposal_samples + self.config_peak_offsets.unsqueeze(0).unsqueeze(1)
        avg_peak_times = avg_peak_times.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        s_new = avg_peak_times + offsets
        left_landmarks = (self.time[torch.cat([self.peak1_left_landmarks, self.peak2_left_landmarks])]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        right_landmarks = (self.time[torch.cat([self.peak1_right_landmarks, self.peak2_right_landmarks])]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        s_new = torch.where(s_new <= left_landmarks, left_landmarks + self.dt, s_new)
        s_new = torch.where(s_new >= right_landmarks, right_landmarks - self.dt, s_new)
        return avg_peak_times, left_landmarks, right_landmarks, s_new


    def compute_warped_times(self, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        landmark_speads = ((right_landmarks - left_landmarks).squeeze()/self.dt).int()
        spike_train_start_offset = torch.searchsorted(self.time, 0, side='left')
        left_shifted_time = [self.time[spike_train_start_offset:(landmark_speads[i] + spike_train_start_offset)] for i in range(landmark_speads.shape[0])]
        left_shifted_peak_times = trial_peak_times - left_landmarks
        right_shifted_peak_times = trial_peak_times - right_landmarks
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        warped_times = [torch.stack([torch.zeros(left_shifted_peak_times.shape[:3], device=self.device)] * left_shifted_time[i].shape[0]) for i in range(landmark_speads.shape[0])]
        max_landmark_spread = landmark_speads.max()
        for i in range(max_landmark_spread):
            for j in range(landmark_speads.shape[0]):
                if i >= landmark_speads[j]: continue
                lst = left_shifted_time[j][i]
                lspt = left_shifted_peak_times[:, :, :, j]
                apt = avg_peak_times[:, :, :, j]
                ls = left_slope[:, :, :, j]
                rs = right_slope[:, :, :, j]
                ll = left_landmarks[:, :, :, j]
                warped_times[j][i] = torch.where(lst < lspt, (lst * ls) + ll, ((lst - lspt) * rs) + apt)
        # warped_times  # 2AL x len(landmark_spread) x N x R X C
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = torch.exp(self.unnormalized_log_factors())
        # warped_time  # len(peak_landmark_spread) x N x R X C X AL
        warped_indices = [warped_times[i]/self.dt for i in range(len(warped_times))]
        floor_warped_indices = [torch.floor(warped_indices[i]).int() for i in range(len(warped_times))]
        ceil_warped_indices = [torch.ceil(warped_indices[i]).int() for i in range(len(warped_times))]
        ceil_weights = [warped_indices[i] - floor_warped_indices[i] for i in range(len(warped_times))]
        floor_weights = [1 - ceil_weights[i] for i in range(len(warped_times))]
        left_landmarks = torch.cat([self.peak1_left_landmarks, self.peak2_left_landmarks])
        right_landmarks = torch.cat([self.peak1_right_landmarks, self.peak2_right_landmarks])
        full_warped_factors = []
        for l in range(self.n_factors):
            floor_warped_factor_l = factors[l, floor_warped_indices[l]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[l]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[l]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[l]
            peak1 = weighted_floor_warped_factor_l + weighted_ceil_warped_factor_l

            floor_warped_factor_l = factors[l, floor_warped_indices[l+self.n_factors]]
            weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[l+self.n_factors]
            ceil_warped_factor_l = factors[l, ceil_warped_indices[l+self.n_factors]]
            weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[l+self.n_factors]
            peak2 = weighted_floor_warped_factor_l + weighted_ceil_warped_factor_l

            early = factors[l, :left_landmarks[l]]
            early = early.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(early.shape[0], *peak1.shape[1:])
            mid = factors[l, right_landmarks[l]:left_landmarks[l+self.n_factors]]
            mid = mid.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(mid.shape[0], *peak1.shape[1:])
            late = factors[l, right_landmarks[l+self.n_factors]:]
            late = late.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(late.shape[0], *peak1.shape[1:])
            full_warped_factors.append(torch.cat([early, peak1, mid, peak2, late], dim=0))
        full_warped_factors = torch.stack(full_warped_factors)
        # full_warped_factors  # AL x T x N x R x C
        return full_warped_factors


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors, posterior_warped_factors):  # and first 2 entropy terms
        # output before summing: C x K x R x L x N
        # factors # L x T
        # warped_factors # L x T x N x R x C
        # posterior_warped_factors # L x T x 1 x R x C
        # neuron_factor_access  # K x L x C
        # Y # K x T x R x C
        K, T, R, C = Y.shape
        neuron_factor_access = neuron_factor_access.permute(2, 0, 1)  # C x K x L
        warped_factors = warped_factors.permute(4, 3, 0, 2, 1)  # C x R x L x N x T
        posterior_warped_factors = posterior_warped_factors.permute(4, 3, 0, 2, 1)  # C x R x L x 1 x T
        Y_sum_t = torch.sum(Y, dim=1).permute(2, 0, 1)  # C x K x R
        Y_sum_rt = torch.sum(Y_sum_t, dim=-1)  # C x K
        alpha = F.softplus(self.alpha)
        log_alpha = torch.log(alpha)

        # M step theta updates
        theta = self.theta_value()
        log_theta = torch.log(theta)
        pi = self.pi_value(neuron_factor_access)
        log_pi = torch.log(pi)
        L_a = self.n_factors // self.n_areas

        # U tensor terms
        # log_posterior_warped_factors_minus_logsumexp_posterior_warped_factors # C x R x L x 1 x T
        log_posterior_warped_factors_minus_logsumexp_posterior_warped_factors = torch.log(posterior_warped_factors) - torch.log(torch.sum(posterior_warped_factors, dim=-1)).unsqueeze(-1)
        # Y_times_posterior_warped_beta  # C x K x L
        Y_times_posterior_warped = torch.einsum('ktrc,crlt->ckl', Y, log_posterior_warped_factors_minus_logsumexp_posterior_warped_factors.squeeze())
        # log_y_factorial_sum_rt # C x K x 1
        log_y_factorial_sum_rt = torch.sum(torch.lgamma(Y + 1), dim=(1,2)).t().unsqueeze(2)
        # Y_sum_rt_times_log_alpha_minus_logsumesp_beta  # C x K x L
        Y_sum_rt_times_log_alpha = torch.einsum('ck,l->ckl', Y_sum_rt, log_alpha)
        # alpha_log_theta  # 1 x 1 x L
        alpha_log_theta = (alpha * log_theta).unsqueeze(0).unsqueeze(1)
        # Y_sum_rt_plus_alpha  # C x K x L
        Y_sum_rt_plus_alpha = Y_sum_rt.unsqueeze(2) + alpha.unsqueeze(0).unsqueeze(1)
        # R_plus_theta  # L
        R_plus_theta = R + theta
        # log_R_plus_theta  # L
        log_R_plus_theta = torch.log(R_plus_theta)
        # Y_sum_rt_plus_alpha_times_log_R_plus_theta  # C x K x L
        Y_sum_rt_plus_alpha_times_log_R_plus_theta = torch.einsum('ckl,l->ckl', Y_sum_rt_plus_alpha, log_R_plus_theta)

        # U_tensor # C x K x L
        U_tensor = (Y_times_posterior_warped - log_y_factorial_sum_rt + Y_sum_rt_times_log_alpha + alpha_log_theta -
                    Y_sum_rt_plus_alpha_times_log_R_plus_theta + log_pi.unsqueeze(0).unsqueeze(1))
        # U_tensor # C x K x A x La
        U_tensor = U_tensor.reshape(*U_tensor.shape[:-1], self.n_areas, L_a)
        # W_CKL # C x K x L
        W_CKL = (neuron_factor_access * F.softmax(U_tensor, dim=-1).reshape(*U_tensor.shape[:-2], self.n_factors)).detach()
        W_CKL = self.validated_W_CKL(W_CKL, neuron_factor_access)
        self.W_CKL = W_CKL  # for finding the posterior clustering probabilities

        # a_CKL  # C x K x L
        a_CKL = (Y_sum_rt_plus_alpha / R_plus_theta).detach()
        self.a_CKL = a_CKL  # for finding the posterior neuron firing rates

        # W_tensor # C x K x R x L x N
        W_tensor = self.n_trial_samples**(-1) * W_CKL.unsqueeze(2).unsqueeze(4)

        # E step theta updates
        theta = self.theta_value()
        log_theta = torch.log(theta)
        pi = self.pi_value(neuron_factor_access)
        # b_CKL  # C x K x L
        b_CKL = (torch.digamma(Y_sum_rt_plus_alpha) - torch.log(R + theta)).detach()

        # Liklelihood Terms (unsqueezed)
        # log_warped_factors # C x R x L x N x T
        log_warped_factors = torch.log(warped_factors)
        # Y_times_warped_beta  # C x K x R x L x N
        Y_times_warped_beta = torch.einsum('ktrc,crlnt->ckrln', Y, log_warped_factors)
        # logsumexp_warped_beta  # C x R x L x N
        logsumexp_warped_beta = torch.logsumexp(log_warped_factors, dim=-1)
        # Y_sum_t_times_logsumexp_warped_beta  # C x K x R x L x N
        Y_sum_t_times_logsumexp_warped_beta = torch.einsum('ckr,crln->ckrln', Y_sum_t, logsumexp_warped_beta)
        #alpha_times_log_theta_plus_b_CKL  # C x K x 1 x L x 1
        alpha_times_log_theta_plus_b_CKL = (alpha * (log_theta + b_CKL)).unsqueeze(2).unsqueeze(4)
        log_gamma_alpha = torch.lgamma(alpha).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)  # 1 x 1 x 1 x L x 1
        # neg_log_P # C x 1 x R x 1 x N
        ltri_matrix = self.ltri_matix()
        neg_log_P = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples, ltri_matrix).unsqueeze(1).unsqueeze(3)
        # neg_log_Q # C x 1 x R x 1 x N
        sd_matrix = self.sd_matrix()
        neg_log_Q = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples - self.trial_peak_offset_proposal_means.unsqueeze(0), sd_matrix).unsqueeze(1).unsqueeze(3)

        elbo = (Y_times_warped_beta - Y_sum_t_times_logsumexp_warped_beta + (1/R) * (alpha_times_log_theta_plus_b_CKL -
                log_gamma_alpha) - (1/K) * (neg_log_P - neg_log_Q))
        elbo = W_tensor * elbo

        return torch.sum(elbo)

    def Sigma_log_likelihood(self, trial_peak_offsets, ltri_matrix):
        n_dims = ltri_matrix.shape[0]  # 2AL
        Sigma = ltri_matrix @ ltri_matrix.t()
        det_Sigma = torch.linalg.det(Sigma)
        inv_Sigma = torch.linalg.inv(Sigma)
        prod_term = torch.einsum('nrcl,lj,nrcj->crn', trial_peak_offsets, inv_Sigma, trial_peak_offsets)  # sum over l
        entropy_term = 0.5 * (n_dims * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_Sigma) + prod_term)
        return entropy_term # C x R x N

    def compute_penalty_terms(self, tau_beta, tau_config, tau_sigma, tau_sd):
        # Penalty Terms
        config_Penalty = - tau_config * torch.sum(self.config_peak_offsets * self.config_peak_offsets)
        proposal_sd_penalty = - tau_sd * torch.sum(self.trial_peak_offset_proposal_sds * self.trial_peak_offset_proposal_sds)
        ltri_matrix = self.ltri_matix()
        Sigma = ltri_matrix @ ltri_matrix.t()
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty = -tau_sigma * (torch.sum(torch.abs(inv_Sigma)) - torch.sum(torch.abs(torch.diag(inv_Sigma))))
        beta_s2_penalty = -tau_beta * torch.sum((self.unnormalized_log_factors() @ self.Delta2TDelta2) * self.unnormalized_log_factors())
        penalty_term = config_Penalty + sigma_Penalty + beta_s2_penalty + proposal_sd_penalty
        return penalty_term


    def infer_latent_variables(self):
        # likelihoods # C x K x L
        C, K, L = torch.where(self.W_CKL == torch.max(self.W_CKL, dim=-1, keepdim=True).values)
        grouped = pd.DataFrame({
            'C': C.cpu(),
            'K': K.cpu(),
            'L': L.cpu()
        }).groupby(['C', 'K']).agg({'L': lambda x: np.random.choice(x)}).reset_index()
        neuron_factor_assignment = torch.zeros_like(self.W_CKL)
        neuron_factor_assignment[grouped['C'], grouped['K'], grouped['L']] = 1
        neuron_firing_rates = torch.sum(self.a_CKL * neuron_factor_assignment, dim=2)
        return neuron_factor_assignment, neuron_firing_rates


    def forward(self, Y, neuron_factor_access, tau_beta, tau_config, tau_sigma, tau_sd):
        self.generate_trial_peak_offset_single_sample()
        posterior_warped_factors = self.warp_all_latent_factors_for_all_trials()
        self.generate_trial_peak_offset_samples()
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors, posterior_warped_factors)
        # penalty terms
        penalty_term = self.compute_penalty_terms(tau_beta, tau_config, tau_sigma, tau_sd)
        return likelihood_term, penalty_term


    def evaluate(self, Y, neuron_factor_access):
        self.generate_trial_peak_offset_single_sample()
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors, warped_factors)
        neuron_factor_assignment, neuron_firing_rates = self.infer_latent_variables()
        return likelihood_term, neuron_factor_assignment, neuron_firing_rates
