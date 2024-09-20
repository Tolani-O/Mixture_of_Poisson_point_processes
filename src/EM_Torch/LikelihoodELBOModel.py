import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix, create_first_diff_matrix
import numpy as np


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trials, n_trial_samples):
        super(LikelihoodELBOModel, self).__init__()

        self.device = 'cpu'
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
        self.n_factors = n_factors
        self.n_areas = n_areas
        self.n_trial_samples = n_trial_samples
        self.n_configs = n_configs
        self.n_trials = n_trials
        Delta2 = create_second_diff_matrix(T)
        # Delta2 = create_first_diff_matrix(T)
        self.Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2)  # T x T # tikhonov regularization

        # Storage for use in the forward pass
        self.W_CKL = None  # C x K x L
        self.W_CRN = None  # C x R x N
        self.trial_peak_offset_proposal_samples = None  # N x R x C x 2AL
        self.a_CKL = None  # C x K x L
        self.theta_init = None  # 1 x AL
        self.pi_init = None  # 1 x AL

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        # self.coupling = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri_diag = None
        self.trial_peak_offset_covar_ltri_offdiag = None
        self.trial_peak_offset_proposal_means = None  # R x C x 2AL
        self.trial_peak_offset_proposal_sds = None  # 2AL
        # self.smoothness_budget = None  # L x 1


    def init_random(self):
        latent_factors = torch.softmax(torch.randn(self.n_factors, self.time.shape[0], dtype=torch.float64), dim=-1)
        self.beta = nn.Parameter(torch.log(latent_factors))
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
        latent_factors = torch.softmax(torch.zeros(self.n_factors, self.time.shape[0], dtype=torch.float64), dim=-1)
        self.beta = nn.Parameter(torch.log(latent_factors))
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
        self.W_CRN = None
        self.a_CKL = None


    def init_ground_truth(self, beta=None, alpha=None, theta=None, pi=None,
                          trial_peak_offset_proposal_means=None,
                          trial_peak_offset_proposal_sds=None,
                          config_peak_offsets=None, trial_peak_offset_covar_ltri=None,
                          init='zeros'):
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
        if pi is not None:
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


    def init_from_data(self, Y, factor_access, sd_init, init='zeros'):
        Y, factor_access = Y.cpu(), factor_access.cpu()
        # Y # K x T x R x C
        # factor_access  # K x L x C
        K, T, R, C = Y.shape
        summed_neurons = torch.einsum('ktrc,klc->lt', Y, factor_access)
        latent_factors = summed_neurons + torch.sum(factor_access, dim=(0, 2)).unsqueeze(1) * torch.rand(self.n_factors, T)
        latent_factors = latent_factors / torch.sum(latent_factors, dim=-1, keepdim=True)
        beta = torch.log(latent_factors)
        spike_counts = torch.einsum('ktrc,klc->krlc', Y, factor_access)
        avg_spike_counts = torch.sum(spike_counts, dim=(0,1,3)) / (R * torch.sum(factor_access, dim=(0, 2)))
        centered_spike_counts = torch.einsum('krlc,klc->krlc', spike_counts - avg_spike_counts.unsqueeze(0).unsqueeze(1).unsqueeze(3), factor_access)
        spike_ct_sd = torch.sqrt(torch.sum(centered_spike_counts**2, dim=(0,1,3)) / (R * K * C))
        alpha = (avg_spike_counts/spike_ct_sd)**2
        theta = avg_spike_counts/(spike_ct_sd**2)
        trial_peak_offset_proposal_sds = sd_init * torch.ones(self.n_factors*self.n_areas, dtype=torch.float64)
        self.init_ground_truth(beta=beta, alpha=alpha, theta=theta,
                               trial_peak_offset_proposal_sds=trial_peak_offset_proposal_sds, init=init)


    def cuda(self, device=None):
        self.device = 'cuda'
        self.time = self.time.cuda(device)
        self.theta_init = self.theta_init.cuda(device)
        self.pi_init = self.pi_init.cuda(device)
        self.Delta2TDelta2 = self.Delta2TDelta2.cuda(device)
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cuda(device)
            self.W_CRN = self.W_CRN.cuda(device)
            self.a_CKL = self.a_CKL.cuda(device)
        super(LikelihoodELBOModel, self).cuda(device)
        return self


    def cpu(self):
        self.device = 'cpu'
        self.time = self.time.cpu()
        self.theta_init = self.theta_init.cpu()
        self.pi_init = self.pi_init.cpu()
        self.Delta2TDelta2 = self.Delta2TDelta2.cpu()
        if self.W_CKL is not None:
            self.W_CKL = self.W_CKL.cpu()
            self.W_CRN = self.W_CRN.cpu()
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

    def theta_value(self):
        if self.W_CKL is None or self.a_CKL is None:
            return self.theta_init
        # W_L  # L
        W_L = torch.sum(self.W_CKL, dim=(0, 1))
        # Wa_L  # L
        Wa_L = torch.sum(self.W_CKL * self.a_CKL, dim=(0, 1))
        theta = self.alpha * (W_L / Wa_L)
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
                                                   torch.einsum('lj,nrcj->nrcl', sd_matrix, gaussian_sample))


    def generate_trial_peak_offset_single_sample(self):
        self.trial_peak_offset_proposal_samples = self.trial_peak_offset_proposal_means.unsqueeze(0)


    def warp_all_latent_factors_for_all_trials(self):
        avg_peak_times, left_landmarks, right_landmarks, s_new = self.compute_offsets_and_landmarks()
        warped_times = self.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_factors = self.compute_warped_factors(warped_times)
        return warped_factors


    def compute_offsets_and_landmarks(self):
        factors = torch.exp(self.beta)
        avg_peak1_times = self.time[self.left_landmark1 + torch.argmax(factors[:, self.left_landmark1:self.right_landmark1], dim=1)]
        avg_peak2_times = self.time[self.left_landmark2 + torch.argmax(factors[:, self.left_landmark2:self.right_landmark2], dim=1)]
        avg_peak_times = torch.cat([avg_peak1_times, avg_peak2_times])
        # avg_peak_times  # 2AL
        # self.trial_peak_offset_proposal_samples  N x R x C x 2AL
        # self.config_peak_offsets  # C x 2AL
        offsets = self.trial_peak_offset_proposal_samples + self.config_peak_offsets.unsqueeze(0).unsqueeze(1)
        avg_peak_times = avg_peak_times.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        s_new = avg_peak_times + offsets
        left_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.left_landmark1, self.left_landmark2]), s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        right_landmarks = (self.time[torch.repeat_interleave(torch.tensor([self.right_landmark1, self.right_landmark2]), s_new.shape[-1] // 2)]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
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
        # warped_time  # 50 x N x R X C X 2AL
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = torch.exp(self.beta)
        # warped_time  # 50 x N x R X C X 2AL
        warped_indices = warped_times / self.dt
        floor_warped_indices = torch.floor(warped_indices).int()
        ceil_warped_indices = torch.ceil(warped_indices).int()
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
        weighted_floor_warped_factors = torch.stack(weighted_floor_warped_factors)
        weighted_ceil_warped_factors = torch.stack(weighted_ceil_warped_factors)
        warped_factors = weighted_floor_warped_factors + weighted_ceil_warped_factors

        # warped_factors  # L x 50 x N x R X C X 2
        early = factors[:, :self.left_landmark1]
        early = early.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(*early.shape,*warped_factors.shape[2:-1])
        mid = factors[:, self.right_landmark1:self.left_landmark2]
        mid = mid.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(*mid.shape,*warped_factors.shape[2:-1])
        late = factors[:, self.right_landmark2:]
        late = late.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(*late.shape,*warped_factors.shape[2:-1])
        warped_factors = torch.cat([early, warped_factors[:, :, :, :, :, 0], mid, warped_factors[:, :, :, :, :, 1], late], dim=1)
        return warped_factors


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors):  # and first 2 entropy terms
        # output before summing: C x K x R x L x N
        # factors # L x T
        # warped_factors # L x T x N x R x C
        # neuron_factor_access  # K x L x C
        # Y # K x T x R x C
        K, T, R, C = Y.shape
        neuron_factor_access = neuron_factor_access.permute(2, 0, 1)  # C x K x L
        warped_factors = warped_factors.permute(4,3,0,2,1)  # C x R x L x N x T
        log_warped_factors = torch.log(warped_factors)  # warped beta # C x R x L x N x T
        beta = self.beta  # beta # L x T
        Y_sum_t = torch.sum(Y, dim=1).permute(2, 0, 1)  # C x K x R
        Y_sum_rt = torch.sum(Y_sum_t, dim=-1)  # C x K
        log_y_factorial = torch.lgamma(Y + 1)  # K x T x R x C
        log_y_factorial_sum_t = torch.sum(log_y_factorial, dim=1).permute(2, 0, 1)  # C x K x R
        log_y_factorial_sum_rt = torch.sum(log_y_factorial_sum_t, dim=-1)  # C x K
        alpha = F.softplus(self.alpha)
        log_alpha = torch.log(alpha)
        # M step theta updates
        theta = self.theta_value()
        log_theta = torch.log(theta)
        pi = self.pi_value(neuron_factor_access)
        log_pi = torch.log(pi)
        alpha_log_theta = alpha * log_theta  # L
        L_a = self.n_factors // self.n_areas

        # U tensor terms
        # Y_times_beta  # C x K x R x L x T
        Y_times_beta = torch.einsum('ktrc,lt->ckrlt', Y, beta)
        # Y_times_beta  # C x K x L
        Y_times_beta = torch.sum(Y_times_beta, dim=(2, 4))
        # log_y_factorial_sum_rt # C x K x 1
        log_y_factorial_sum_rt = log_y_factorial_sum_rt.unsqueeze(2)
        # log_alpha_minus_logsumesp_beta  # L
        log_alpha_minus_logsumesp_beta = log_alpha - torch.logsumexp(beta, dim=-1)
        # Y_sum_rt_times_log_alpha_minus_logsumesp_beta  # C x K x L
        Y_sum_rt_times_log_alpha_minus_logsumesp_beta = torch.einsum('ck,l->ckl', Y_sum_rt, log_alpha_minus_logsumesp_beta)
        # Y_sum_rt_plus_alpha  # C x K x L
        Y_sum_rt_plus_alpha = Y_sum_rt.unsqueeze(2) + alpha.unsqueeze(0).unsqueeze(1)
        # R_plus_theta  # L
        R_plus_theta = R + theta
        # log_R_plus_theta  # L
        log_R_plus_theta = torch.log(R_plus_theta)
        # Y_sum_rt_plus_alpha_times_log_R_plus_theta  # C x K x L
        Y_sum_rt_plus_alpha_times_log_R_plus_theta = torch.einsum('ckl,l->ckl', Y_sum_rt_plus_alpha, log_R_plus_theta)

        # U_tensor # C x K x L
        U_tensor = (Y_times_beta - log_y_factorial_sum_rt + Y_sum_rt_times_log_alpha_minus_logsumesp_beta + alpha_log_theta.unsqueeze(0).unsqueeze(1) -
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

        # V tensor terms
        # Y_times_warped_beta  # C x K x R x L x N x T
        Y_times_warped_beta = torch.einsum('ktrc,crlnt->ckrlnt', Y, log_warped_factors)
        # Y_times_warped_beta  # C x K x R x L x N
        Y_times_warped_beta = torch.sum(Y_times_warped_beta, dim=-1)
        # log_y_factorial_sum_t # C x K x R x 1 x 1
        log_y_factorial_sum_t = log_y_factorial_sum_t.unsqueeze(3).unsqueeze(4)
        # logsumexp_warped_beta  # C x R x L x N
        logsumexp_warped_beta = torch.logsumexp(log_warped_factors, dim=-1)
        # log_alpha_minus_logsumesp_warped_beta  # # C x R x L x N
        log_alpha_minus_logsumesp_warped_beta = log_alpha.unsqueeze(0).unsqueeze(1).unsqueeze(3) - logsumexp_warped_beta
        # Y_sum_t_times_log_alpha_minus_logsumesp_log_warped_factors  # C x K x R x L x N
        Y_sum_t_times_log_alpha_minus_logsumesp_log_warped_factors = torch.einsum('ckr,crln->ckrln', Y_sum_t, log_alpha_minus_logsumesp_warped_beta)
        # Y_sum_t_plus_alpha  # C x K x R x L
        Y_sum_t_plus_alpha = Y_sum_t.unsqueeze(3) + alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # one_plus_theta  # L
        one_plus_theta = 1 + theta
        # log_one_plus_theta  # L
        log_one_plus_theta = torch.log(one_plus_theta)
        # Y_sum_t_plus_alpha_times_log_one_plus_theta  # C x K x R x L
        Y_sum_t_plus_alpha_times_log_one_plus_theta = torch.einsum('ckrl,l->ckrl', Y_sum_t_plus_alpha, log_one_plus_theta)

        # V_tensor # C x K x R x L x N
        V_tensor = (Y_times_warped_beta - log_y_factorial_sum_t + Y_sum_t_times_log_alpha_minus_logsumesp_log_warped_factors +
                    alpha_log_theta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4) -
                    Y_sum_t_plus_alpha_times_log_one_plus_theta.unsqueeze(4) +
                    log_pi.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
        # V_tensor # C x K x R x A x La x N
        V_tensor = V_tensor.reshape(*V_tensor.shape[:3], self.n_areas, L_a, V_tensor.shape[-1])
        # logsumexp_V_tensor # C x K x R x A x N
        logsumexp_V_tensor = torch.logsumexp(V_tensor, dim=-2)
        # neuron_area_access  #  C x K x 1 x A x 1
        neuron_area_access = neuron_factor_access[:, :, [i * L_a for i in range(self.n_areas)]].unsqueeze(2).unsqueeze(4)
        # sum_k_logsumexp_V_tensor # C x R x N
        sum_k_logsumexp_V_tensor = torch.sum(logsumexp_V_tensor * neuron_area_access, dim=(1, 3))

        # VV tensor terms
        ltri_matrix = self.ltri_matix()
        sd_matrix = self.sd_matrix()
        # neg_log_P # C x R x N
        neg_log_P = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples, ltri_matrix)
        # neg_log_Q # C x R x N
        neg_log_Q = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples - self.trial_peak_offset_proposal_means.unsqueeze(0), sd_matrix)

        # VV tensor # C x R x N
        VV_tensor = neg_log_Q - (neg_log_P - sum_k_logsumexp_V_tensor).detach()
        # W_CRN # C x R x N
        W_CRN = F.softmax(VV_tensor, dim=-1).detach()
        self.W_CRN = W_CRN

        # W_tensor # C x K x R x L x N
        W_tensor = W_CKL.unsqueeze(2).unsqueeze(4) * W_CRN.unsqueeze(1).unsqueeze(3)

        # E step theta updates
        theta = self.theta_value()
        log_theta = torch.log(theta)
        pi = self.pi_value(neuron_factor_access)
        # b_CKL  # C x K x L
        b_CKL = (torch.digamma(Y_sum_rt_plus_alpha) - torch.log(R + theta)).detach()

        # Liklelihood Terms (unsqueezed)
        # Y_sum_t_times_logsumexp_warped_beta  # C x K x R x L x N
        Y_sum_t_times_logsumexp_warped_beta = torch.einsum('ckr,crln->ckrln', Y_sum_t, logsumexp_warped_beta)
        #alpha_times_log_theta_plus_b_CKL  # C x K x 1 x L x 1
        alpha_times_log_theta_plus_b_CKL = (alpha * (log_theta + b_CKL)).unsqueeze(2).unsqueeze(4)
        log_gamma_alpha = torch.lgamma(alpha).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)  # 1 x 1 x 1 x L x 1
        # neg_log_P # C x 1 x R x 1 x N
        neg_log_P = neg_log_P.unsqueeze(1).unsqueeze(3)
        # logsumexp_VV_tensor #  C x 1 x R x 1 x 1
        logsumexp_VV_tensor = torch.logsumexp(VV_tensor, dim=-1).unsqueeze(1).unsqueeze(3).unsqueeze(4)

        elbo = (Y_times_warped_beta - Y_sum_t_times_logsumexp_warped_beta + (1/R) * (alpha_times_log_theta_plus_b_CKL -
                log_gamma_alpha) - (1/K) * (neg_log_P - logsumexp_VV_tensor))
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
        beta_s2_penalty = -tau_beta * torch.sum((self.beta @ self.Delta2TDelta2) * self.beta)
        penalty_term = config_Penalty + sigma_Penalty + beta_s2_penalty + proposal_sd_penalty
        return penalty_term


    def infer_latent_variables(self):
        # likelihoods # C x K x L
        neuron_factor_assignment = torch.where(self.W_CKL == torch.max(self.W_CKL, dim=-1, keepdim=True).values, 1, 0)
        neuron_firing_rates = torch.sum(self.a_CKL * neuron_factor_assignment, dim=2)
        effective_sample_size = torch.sum(self.W_CRN**2, dim=-1)**(-1)
        # trial_peak_offset  R x C x 2AL
        trial_peak_offsets = torch.einsum('nrcl,crn->crl', self.trial_peak_offset_proposal_samples, self.W_CRN)
        return neuron_factor_assignment, neuron_firing_rates, effective_sample_size, trial_peak_offsets


    def forward(self, Y, neuron_factor_access, tau_beta, tau_config, tau_sigma, tau_sd, stage=1):
        self.beta.requires_grad = False
        self.alpha.requires_grad = False
        self.config_peak_offsets.requires_grad = False
        self.trial_peak_offset_covar_ltri_diag.requires_grad = False
        self.trial_peak_offset_covar_ltri_offdiag.requires_grad = False
        self.trial_peak_offset_proposal_means.requires_grad = False
        self.trial_peak_offset_proposal_sds.requires_grad = False
        if stage == 1:
            # update beta and alpha
            self.beta.requires_grad = True
            self.alpha.requires_grad = True
        elif stage == 2:
            # update config_peak_offsets and ltri maxtrix
            self.config_peak_offsets.requires_grad = True
            self.trial_peak_offset_covar_ltri_diag.requires_grad = True
            self.trial_peak_offset_covar_ltri_offdiag.requires_grad = True
        else:
            # update trial_peak_offset_proposal_means and trial_peak_offset_proposal_sds
            self.trial_peak_offset_proposal_means.requires_grad = True
            self.trial_peak_offset_proposal_sds.requires_grad = True
        self.generate_trial_peak_offset_samples()
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
        # penalty terms
        penalty_term = self.compute_penalty_terms(tau_beta, tau_config, tau_sigma, tau_sd)
        return likelihood_term, penalty_term


    def evaluate(self, Y, neuron_factor_access):
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
        neuron_factor_assignment, neuron_firing_rates, effective_sample_size, trial_peak_offsets = self.infer_latent_variables()
        return likelihood_term, neuron_factor_assignment, neuron_firing_rates, effective_sample_size, trial_peak_offsets
