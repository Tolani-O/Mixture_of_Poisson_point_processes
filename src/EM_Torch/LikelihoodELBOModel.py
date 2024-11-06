import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_first_diff_matrix, create_second_diff_matrix
import numpy as np
import pandas as pd
import pickle


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trials, n_trial_samples,
                 peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                 temperature=None, weights=None):
        super(LikelihoodELBOModel, self).__init__()

        self.device = 'cpu'
        self.is_eval = True
        if temperature is None:
            temperature = (1,)
        if isinstance(temperature, (int, float)):
            temperature = (temperature,)
        if weights is None:
            weights = [1] * len(temperature)
        if isinstance(weights, (int, float)):
            weights = [weights] * len(temperature)
        assert len(temperature) == len(weights), "Temperature and weights must be the same length"
        self.temperature = torch.tensor(temperature)
        self.weights = torch.tensor(weights, dtype=torch.float)
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
        self.trial_peak_offset_proposal_samples = None  # N x R x C x 2AL
        # There are not learned but computed in the forward pass
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.W_CKL = None  # C x K x L
        self.a_CKL = None  # C x K x L
        self.log_sum_exp_U_tensor = None  # C x K x L

        # Parameters
        self.beta = None  # AL x T
        self.alpha = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri_diag = None
        self.trial_peak_offset_covar_ltri_offdiag = None
        self.trial_peak_offset_proposal_means = None  # R x C x 2AL
        self.trial_peak_offset_proposal_sds = None  # R x C x 2AL


    def init_random(self):
        self.beta = nn.Parameter(torch.log(torch.randn(self.n_factors, self.time.shape[0], dtype=torch.float64)))
        self.alpha = nn.Parameter(torch.randn(self.n_factors, dtype=torch.float64))
        n_dims = 2 * self.n_factors
        num_elements = n_dims * (n_dims - 1) // 2
        self.config_peak_offsets = nn.Parameter(torch.randn(self.n_configs, n_dims, dtype=torch.float64))
        self.trial_peak_offset_covar_ltri_diag = nn.Parameter(torch.rand(n_dims, dtype=torch.float64)+1)
        self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(torch.randn(num_elements, dtype=torch.float64))
        self.trial_peak_offset_proposal_means = nn.Parameter(torch.randn(self.n_trials, self.n_configs, n_dims, dtype=torch.float64))
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.rand(self.n_trials, self.n_configs, n_dims, dtype=torch.float64) + 1)
        self.pi = F.softmax(torch.randn(self.n_areas, self.n_factors // self.n_areas, dtype=torch.float64), dim=1).flatten()
        self.standard_init()


    def init_zero(self):
        self.beta = nn.Parameter(torch.zeros(self.n_factors, self.time.shape[0], dtype=torch.float64))
        self.alpha = nn.Parameter(torch.ones(self.n_factors, dtype=torch.float64))
        n_dims = 2 * self.n_factors
        num_elements = n_dims * (n_dims - 1) // 2
        self.config_peak_offsets = nn.Parameter(torch.zeros(self.n_configs, n_dims, dtype=torch.float64))
        self.trial_peak_offset_covar_ltri_diag = nn.Parameter(torch.ones(n_dims, dtype=torch.float64))
        self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(torch.zeros(num_elements, dtype=torch.float64))
        self.trial_peak_offset_proposal_means = nn.Parameter(torch.zeros(self.n_trials, self.n_configs, n_dims, dtype=torch.float64))
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.ones(self.n_trials, self.n_configs, n_dims, dtype=torch.float64))
        self.pi = F.softmax(torch.zeros(self.n_areas, self.n_factors // self.n_areas, dtype=torch.float64), dim=1).flatten()
        self.standard_init()


    def standard_init(self):
        self.theta = None
        self.W_CKL = None
        self.a_CKL = None


    def init_ground_truth(self, beta=None, alpha=None, theta=None, pi=None,
                          trial_peak_offset_proposal_means=None, sd_init=None,
                          config_peak_offsets=None, trial_peak_offset_covar_ltri=None,
                          W_CKL=None, init='zeros'):
        n_dims = 2 * self.n_factors
        if init == 'zeros':
            self.init_zero()
        elif init == 'random':
            self.init_random()
        if beta is not None:
            self.beta = nn.Parameter(beta)
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
        if theta is not None:
            self.theta = theta
        if W_CKL is not None:
            self.validated_W_CKL(W_CKL)
        elif pi is not None:
            self.pi = pi
        if trial_peak_offset_proposal_means is not None:
            self.trial_peak_offset_proposal_means = nn.Parameter(trial_peak_offset_proposal_means)
        if sd_init is not None:
            self.trial_peak_offset_proposal_sds = nn.Parameter(sd_init * torch.ones(self.n_trials, self.n_configs, n_dims,
                                                                                    dtype=torch.float64, device=self.device))
        if config_peak_offsets is not None:
            self.config_peak_offsets = nn.Parameter(config_peak_offsets)
        if trial_peak_offset_covar_ltri is not None:
            self.trial_peak_offset_covar_ltri_diag = nn.Parameter(trial_peak_offset_covar_ltri.diag())
            indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
            self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(trial_peak_offset_covar_ltri[indices[0], indices[1]])


    def init_from_data(self, Y, factor_access, sd_init, cluster_dir=None, init='zeros'):
        # Y # K x T x R x C
        # factor_access  # C x K x L
        _, T, R, _ = Y.shape
        if cluster_dir is None:
            W_CKL = None
            pi = None
            summed_neurons = torch.einsum('ktrc,ckl->lt', Y, factor_access)
            latent_factors = summed_neurons + torch.sqrt(torch.sum(summed_neurons, dim=-1)).unsqueeze(1) * torch.rand(self.n_factors, T)
            beta = torch.log(latent_factors)
        else:
            cluster_dir = os.path.join(cluster_dir, 'cluster_initialization.pkl')
            if not os.path.exists(cluster_dir):
                raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
            print('Loading clusters from: ', cluster_dir)
            with open(cluster_dir, 'rb') as f:
                data = pickle.load(f)
            W_CKL, beta = data['neuron_factor_assignment'], data['beta']
            W_L = torch.sum(W_CKL, dim=(0, 1))
            pi = W_L / torch.sum(factor_access, dim=(0, 1))
        spike_counts = torch.einsum('ktrc,ckl->krlc', Y, factor_access)
        avg_spike_counts = torch.sum(spike_counts, dim=(0,1,3)) / (R * torch.sum(factor_access, dim=(0, 1)))
        print('Average spike counts:')
        print(avg_spike_counts.reshape(self.n_areas, -1).numpy())
        centered_spike_counts = torch.einsum('krlc,ckl->krlc', spike_counts - avg_spike_counts.unsqueeze(0).unsqueeze(1).unsqueeze(3), factor_access)
        spike_ct_var = torch.sum(centered_spike_counts**2, dim=(0,1,3)) / ((R * torch.sum(factor_access, dim=(0, 1)))-1)
        print('Spike count variance - Average spike counts:')
        print((spike_ct_var-avg_spike_counts).reshape(self.n_areas, -1).numpy())
        alpha = (avg_spike_counts)**2/(spike_ct_var-avg_spike_counts)
        alpha = alpha.expm1().clamp_min(1e-6).log()
        theta = avg_spike_counts/(spike_ct_var-avg_spike_counts)
        self.init_ground_truth(beta=beta, alpha=alpha, theta=theta, sd_init=sd_init, W_CKL=W_CKL, pi=pi, init=init)


    # move to cuda flag tells the function whether gpus are available
    def cuda(self, device=None, move_to_cuda=True):
        if (not move_to_cuda) or (self.device == 'cuda'):
            return
        self.device = 'cuda'
        self.time = self.time.cuda(device)
        self.temperature = self.temperature.cuda(device)
        self.weights = self.weights.cuda(device)
        self.Delta2TDelta2 = self.Delta2TDelta2.cuda(device)
        self.peak1_left_landmarks = self.peak1_left_landmarks.cuda(device)
        self.peak2_left_landmarks = self.peak2_left_landmarks.cuda(device)
        self.peak1_right_landmarks = self.peak1_right_landmarks.cuda(device)
        self.peak2_right_landmarks = self.peak2_right_landmarks.cuda(device)
        if self.theta is not None:
            self.theta = self.theta.cuda(device)
        if self.pi is not None:
            self.pi = self.pi.cuda(device)
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cuda(device)
        if self.a_CKL is not None:
            self.a_CKL = self.a_CKL.cuda(device)
        super(LikelihoodELBOModel, self).cuda(device)


    def cpu(self):
        if self.device == 'cpu':
            return
        self.device = 'cpu'
        self.time = self.time.cpu()
        self.temperature = self.temperature.cpu()
        self.weights = self.weights.cpu()
        self.Delta2TDelta2 = self.Delta2TDelta2.cpu()
        self.peak1_left_landmarks = self.peak1_left_landmarks.cpu()
        self.peak2_left_landmarks = self.peak2_left_landmarks.cpu()
        self.peak1_right_landmarks = self.peak1_right_landmarks.cpu()
        self.peak2_right_landmarks = self.peak2_right_landmarks.cpu()
        if self.theta is not None:
            self.theta = self.theta.cpu()
        if self.pi is not None:
            self.pi = self.pi.cpu()
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cpu()
        if self.a_CKL is not None:
            self.a_CKL = self.a_CKL.cpu()
        super(LikelihoodELBOModel, self).cpu()


    def train(self, mode=True):
        if self.is_eval != mode:
            return
        self.is_eval = not mode
        super(LikelihoodELBOModel, self).train(mode)


    def ltri_matix(self, device=None):
        if device is None:
            device = self.device
        n_dims = 2 * self.n_factors
        ltri_matrix = torch.zeros(n_dims, n_dims, dtype=torch.float64, device=device)
        ltri_matrix[torch.arange(n_dims), torch.arange(n_dims)] = self.trial_peak_offset_covar_ltri_diag
        indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
        ltri_matrix[indices[0], indices[1]] = self.trial_peak_offset_covar_ltri_offdiag
        return ltri_matrix


    def unnormalized_log_factors(self):
        return self.beta - self.beta[:, 0].unsqueeze(1).expand_as(self.beta)


    def update_params(self, Y_sum_rt_plus_alpha, neuron_factor_access, R):
        if self.W_CKL is None:
            return
        self.a_CKL = (Y_sum_rt_plus_alpha / (R + self.theta)).detach()
        # W_L  # L
        W_L = torch.sum(self.W_CKL, dim=(0, 1))
        pi = W_L / torch.sum(neuron_factor_access, dim=(0, 1))
        self.pi = pi.detach()
        # Wa_L  # L
        Wa_L = torch.sum(self.W_CKL * self.a_CKL, dim=(0, 1))
        theta = F.softplus(self.alpha) * (W_L / Wa_L)
        self.theta = theta.detach()


    def validated_W_CKL(self, W_CKL, tol=1e-10):
        W_L = torch.sum(W_CKL, dim=(0, 1)).reshape(self.n_areas, -1)
        W_CKL_L = W_CKL.reshape(W_CKL.shape[0], W_CKL.shape[1], self.n_areas, -1)
        for i in range(self.n_areas):
            while torch.any(W_L[i] < tol):
                min_idx = torch.argmin(W_L[i])  # index of min population factor
                max_idx = torch.argmax(W_L[i])  # index of max population factor
                highest_assignmet = torch.max(W_CKL_L[:, :, i, max_idx])  # highest assignment to max factor
                max_members_idx = torch.where(W_CKL_L[:, :, i, max_idx] == highest_assignmet)
                singele_max_member_idx = [max_members_idx[0][0], max_members_idx[1][0]]
                frac_to_move = tol if highest_assignmet/1000 > tol else highest_assignmet/1000
                W_CKL_L[singele_max_member_idx[0], singele_max_member_idx[1], i, max_idx] -= frac_to_move
                W_CKL_L[singele_max_member_idx[0], singele_max_member_idx[1], i, min_idx] += frac_to_move
                W_L = torch.sum(W_CKL_L, dim=(0, 1))
        self.W_CKL = W_CKL_L.reshape(*W_CKL.shape[:2], -1)


    def generate_trial_peak_offset_samples(self):
        if self.is_eval:
            # trial_peak_offset_proposal_samples 1 x R x C x 2AL
            self.trial_peak_offset_proposal_samples = self.trial_peak_offset_proposal_means.unsqueeze(0)
        else:
            gaussian_sample = torch.concat([torch.randn(self.n_trial_samples, self.n_trials, self.n_configs, 2 * self.n_factors,
                                                        device=self.device, dtype=torch.float64),
                                            torch.zeros(1, self.n_trials, self.n_configs, 2 * self.n_factors,
                                                        device=self.device, dtype=torch.float64)], dim=0)
            # trial_peak_offset_proposal_samples 1+N x R x C x 2AL
            self.trial_peak_offset_proposal_samples = (self.trial_peak_offset_proposal_means.unsqueeze(0) +
                                                       gaussian_sample * self.trial_peak_offset_proposal_sds.unsqueeze(0))


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
        landmark_speads = ((right_landmarks - left_landmarks).squeeze()/self.dt).round().int()
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


    def prepare_inputs(self, processed_inputs):
        self.generate_trial_peak_offset_samples()
        # warped_factors # L x T x N x R x C --> C x R x L x N x T
        all_warped_factors = self.warp_all_latent_factors_for_all_trials().permute(4, 3, 0, 2, 1)
        end = torch.max(torch.tensor([all_warped_factors.shape[3]-1, 1], device=self.device)).int()
        warped_factors = all_warped_factors[:, :, :, :end, :]
        posterior_warped_factors = all_warped_factors[:, :, :, -1:, :]
        self.trial_peak_offset_proposal_samples = self.trial_peak_offset_proposal_samples[:end]
        alpha = F.softplus(self.alpha)
        Y_sum_rt = processed_inputs['Y_sum_rt']
        Y_sum_rt_plus_alpha = Y_sum_rt.unsqueeze(2) + alpha.unsqueeze(0).unsqueeze(1)  # C x K x L
        processed_inputs['alpha'] = alpha
        processed_inputs['Y_sum_rt_plus_alpha'] = Y_sum_rt_plus_alpha
        processed_inputs['warped_factors'] = warped_factors
        processed_inputs['posterior_warped_factors'] = posterior_warped_factors
        return processed_inputs


    def E_step_posterior_updates(self, processed_inputs):
        # processed_inputs['Y'] = Y
        # processed_inputs['Y_sum_t'] = Y_sum_t
        # processed_inputs['Y_sum_rt'] = Y_sum_rt
        # processed_inputs['neuron_factor_access'] = neuron_factor_access
        # processed_inputs['alpha'] = alpha
        # processed_inputs['Y_sum_rt_plus_alpha'] = Y_sum_rt_plus_alpha
        # processed_inputs['warped_factors'] = warped_factors
        # processed_inputs['posterior_warped_factors'] = posterior_warped_factors
        Y = processed_inputs['Y']
        Y_sum_rt = processed_inputs['Y_sum_rt']
        Y_sum_rt_plus_alpha = processed_inputs['Y_sum_rt_plus_alpha']
        alpha = processed_inputs['alpha']
        posterior_warped_factors = processed_inputs['posterior_warped_factors']
        neuron_factor_access = processed_inputs['neuron_factor_access']
        # output before summing: C x K x R x L x N
        # factors # L x T
        # posterior_warped_factors # C x R x L x 1 x T
        # neuron_factor_access  # C x K x L
        # Y # K x T x R x C
        K, T, R, C = Y.shape

        ## previous M step parameter updates (for use in current E step)
        self.update_params(Y_sum_rt_plus_alpha, neuron_factor_access, R)

        ## E step posterior computation
        # U tensor terms
        # log_posterior_warped_factors_minus_logsumexp_posterior_warped_factors # C x R x L x 1 x T
        log_posterior_warped_factors_minus_logsum_t_posterior_warped_factors = torch.log(posterior_warped_factors) - torch.log(torch.sum(posterior_warped_factors, dim=-1)).unsqueeze(-1)
        # Y_times_posterior_warped_beta  # C x K x L
        Y_times_posterior_warped = torch.einsum('ktrc,crlt->ckl', Y, log_posterior_warped_factors_minus_logsum_t_posterior_warped_factors.squeeze())
        # log_y_factorial_sum_rt # C x K x 1
        log_y_factorial_sum_rt = torch.sum(torch.lgamma(Y + 1), dim=(1,2)).t().unsqueeze(2)
        grid_y, grid_x = torch.meshgrid(torch.arange(C), torch.arange(K), indexing='ij')
        indx_tracker = Y_sum_rt.flatten().int()
        min_indx = torch.min(indx_tracker)
        max_indx = torch.max(indx_tracker)
        range_indices = torch.arange(max_indx - min_indx, device=self.device)
        last_dim_indices = (indx_tracker.unsqueeze(1) + range_indices.unsqueeze(0)).clamp(max=max_indx-1).t().flatten()
        i_vals = torch.arange(max_indx, device=self.device).unsqueeze(0).unsqueeze(1).expand(C, K, -1)  # C x K x I
        alpha_plus_i_minus_1 = alpha.unsqueeze(0).unsqueeze(1).unsqueeze(3) + i_vals.unsqueeze(2)
        alpha_plus_i_minus_1[grid_y.flatten().repeat(max_indx - min_indx),
                                        grid_x.flatten().repeat(max_indx - min_indx), :,
                                        last_dim_indices] = 1
        # sum_over_y_log_alpha_plus_i_minus_1  # C x K x L
        sum_over_y_log_alpha_plus_i_minus_1 = torch.log(alpha_plus_i_minus_1).sum(dim=-1)
        # alpha_log_theta  # 1 x 1 x L
        alpha_log_theta = (alpha * torch.log(self.theta)).unsqueeze(0).unsqueeze(1)
        # Y_sum_rt_plus_alpha_times_log_R_plus_theta  # C x K x L
        Y_sum_rt_plus_alpha_times_log_R_plus_theta = torch.einsum('ckl,l->ckl', Y_sum_rt_plus_alpha, torch.log(R + self.theta))

        # U_tensor # C x K x L
        L_a = self.n_factors // self.n_areas
        if self.is_eval:
            pi = self.pi
        else:
            temp = self.temperature[torch.multinomial(self.weights, 1)].item()
            pi = F.softmax(torch.log(self.pi).reshape(self.n_areas, L_a) / temp, dim=-1).flatten()
        U_tensor = (Y_times_posterior_warped - log_y_factorial_sum_rt + sum_over_y_log_alpha_plus_i_minus_1 + alpha_log_theta -
                    Y_sum_rt_plus_alpha_times_log_R_plus_theta + torch.log(pi).unsqueeze(0).unsqueeze(1))
        # U_tensor # C x K x A x La
        U_tensor = U_tensor.reshape(*U_tensor.shape[:-1], self.n_areas, L_a)
        neuron_area_access = neuron_factor_access.reshape(*neuron_factor_access.shape[:-1], self.n_areas, L_a).sum(dim=-1)/L_a
        self.log_sum_exp_U_tensor = neuron_area_access * torch.logsumexp(U_tensor, dim=-1)
        # W_CKL # C x K x L
        W_CKL = neuron_factor_access * F.softmax(U_tensor, dim=-1).reshape(*U_tensor.shape[:-2], self.n_factors).detach()
        self.validated_W_CKL(W_CKL)  # for finding the posterior clustering probabilities

        ## current E step parameter updates (for use in next M step)
        self.update_params(Y_sum_rt_plus_alpha, neuron_factor_access, R)


    def ELBO_term(self, processed_inputs):
        Y = processed_inputs['Y']
        Y_sum_t = processed_inputs['Y_sum_t']
        Y_sum_rt_plus_alpha = processed_inputs['Y_sum_rt_plus_alpha']
        alpha = processed_inputs['alpha']
        warped_factors = processed_inputs['warped_factors']
        # warped_factors # C x R x L x N x T
        K, T, R, C = Y.shape
        theta = self.theta
        # b_CKL  # C x K x L
        b_CKL = (torch.digamma(Y_sum_rt_plus_alpha) - torch.log(R + theta)).detach()

        # W_tensor # C x K x 1 x L x 1
        W_tensor = warped_factors.shape[3]**(-1) * self.W_CKL.unsqueeze(2).unsqueeze(4)

        # Liklelihood Terms (unsqueezed)
        # log_warped_factors # C x R x L x N x T
        log_warped_factors = torch.log(warped_factors)
        # Y_times_warped_beta  # C x K x R x L x N
        Y_times_warped_beta = torch.einsum('ktrc,crlnt->ckrln', Y, log_warped_factors)
        # logsumexp_warped_beta  # C x R x L x N
        logsumexp_warped_beta = torch.logsumexp(log_warped_factors, dim=-1)
        # Y_sum_t_times_logsumexp_warped_beta  # C x K x R x L x N
        Y_sum_t_times_logsumexp_warped_beta = torch.einsum('ckr,crln->ckrln', Y_sum_t, logsumexp_warped_beta)
        # alpha_times_log_theta_plus_b_CKL  # C x K x 1 x L x 1
        alpha_times_log_theta_plus_b_CKL = (alpha * (torch.log(theta) + b_CKL)).unsqueeze(2).unsqueeze(4)
        log_gamma_alpha = torch.lgamma(alpha).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)  # 1 x 1 x 1 x L x 1
        # neg_log_P # C x 1 x R x 1 x N
        neg_log_P = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples, self.ltri_matix()).unsqueeze(1).unsqueeze(3)
        # neg_log_Q # C x 1 x R x 1 x N
        neg_log_Q = self.sd_log_likelihood(self.trial_peak_offset_proposal_samples).unsqueeze(1).unsqueeze(3)
        elbo = (Y_times_warped_beta - Y_sum_t_times_logsumexp_warped_beta + (1 / R) * (alpha_times_log_theta_plus_b_CKL -
                    log_gamma_alpha) - (1 / K) * (neg_log_P - neg_log_Q))
        elbo = torch.sum(W_tensor * elbo)
        return elbo


    def Sigma_log_likelihood(self, trial_peak_offsets, ltri_matrix):
        n_dims = ltri_matrix.shape[0]  # 2AL
        Sigma = ltri_matrix @ ltri_matrix.t()
        det_Sigma = torch.linalg.det(Sigma)
        inv_Sigma = torch.linalg.inv(Sigma)
        prod_term = torch.einsum('nrcl,lj,nrcj->crn', trial_peak_offsets, inv_Sigma, trial_peak_offsets)  # sum over l
        entropy_term = 0.5 * (n_dims * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_Sigma) + prod_term)
        return entropy_term # C x R x N


    def sd_log_likelihood(self, trial_peak_offsets):
        trial_peak_offsets = (trial_peak_offsets - self.trial_peak_offset_proposal_means.unsqueeze(0))**2
        n_dims = self.trial_peak_offset_proposal_sds.shape[-1]
        det_Sigma = torch.prod(self.trial_peak_offset_proposal_sds**2, dim=-1)
        inv_Sigma = self.trial_peak_offset_proposal_sds**(-2)
        prod_term = torch.sum(trial_peak_offsets * inv_Sigma.unsqueeze(0), dim=-1).permute(2, 1, 0) # sum over l
        entropy_term = 0.5 * (n_dims * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_Sigma.t().unsqueeze(-1)) + prod_term)
        return entropy_term  # C x R x N


    def compute_penalty_terms(self, tau_beta, tau_config, tau_sigma, tau_sd):
        # Penalty Terms
        config_Penalty = - tau_config * (1/torch.prod(torch.tensor(self.config_peak_offsets.shape))) * torch.sum(self.config_peak_offsets * self.config_peak_offsets)
        proposal_sd_penalty = - tau_sd * (1/torch.prod(torch.tensor(self.trial_peak_offset_proposal_sds.shape))) * torch.sum(self.trial_peak_offset_proposal_sds * self.trial_peak_offset_proposal_sds)
        ltri_matrix = self.ltri_matix()
        Sigma = ltri_matrix @ ltri_matrix.t()
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty = -tau_sigma * (1/(torch.prod(torch.tensor(Sigma.shape))-Sigma.shape[0])) * (torch.sum(torch.abs(inv_Sigma)) - torch.sum(torch.abs(torch.diag(inv_Sigma))))
        factors = torch.softmax(self.unnormalized_log_factors(), dim=-1)
        factor_first_deriv_access = torch.zeros_like(factors)
        L_a = self.n_factors // self.n_areas
        factor_first_deriv_access[[i*L_a for i in range(self.n_areas)], :] = 1
        beta_s2_penalty = -tau_beta * (1/torch.prod(torch.tensor(factors.shape))) * torch.sum((factors @ self.Delta2TDelta2) * factors)
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


    def forward(self, processed_inputs, update_membership=True, train=True):
        self.train(train)
        processed_inputs = self.prepare_inputs(processed_inputs)
        if update_membership:
            self.E_step_posterior_updates(processed_inputs)
        return self.ELBO_term(processed_inputs)


    def log_likelihood(self, processed_inputs):
        self.train(False)
        # trial_peak_offset_proposal_samples 1 x R x C x 2AL
        processed_inputs = self.prepare_inputs(processed_inputs)
        self.E_step_posterior_updates(processed_inputs)
        # log_P  C x R
        log_P = -self.Sigma_log_likelihood(self.trial_peak_offset_proposal_means.unsqueeze(0), self.ltri_matix()).squeeze()
        log_likelihood = torch.sum(self.log_sum_exp_U_tensor) + torch.sum(log_P)
        return log_likelihood
