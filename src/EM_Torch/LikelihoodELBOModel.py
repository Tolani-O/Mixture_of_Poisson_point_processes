import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix
import numpy as np
from scipy.ndimage import gaussian_filter1d


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trial_samples):
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
        self.n_factors = n_factors
        self.n_areas = n_areas
        self.n_trial_samples = n_trial_samples
        self.n_configs = n_configs
        Delta2 = create_second_diff_matrix(T)
        self.Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2).to_sparse()  # T x T # tikhonov regularization

        # Storage for use in the forward pass
        self.neuron_factor_access = None
        self.trial_peak_offsets = None  # NRC x 2AL
        self.W_CKL = None  # C x K x L
        self.W_CRN = None  # C x R x N
        self.a_CKL = None  # C x K x L

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL
        self.smoothness_budget = None  # L x 1


    def init_random(self):
        self.beta = nn.Parameter(torch.randn(self.n_factors, self.time.shape[0]))
        self.alpha = nn.Parameter(torch.randn(self.n_factors))
        self.theta = nn.Parameter(torch.randn(self.n_factors))
        self.pi = nn.Parameter(torch.randn(self.n_areas, self.n_factors//self.n_areas-1))
        self.config_peak_offsets = nn.Parameter(torch.randn(self.n_configs, 2 * self.n_factors))
        matrix = torch.tril(torch.randn(2 * self.n_factors, 2 * self.n_factors))
        # Ensure diagonal elements are positive
        for i in range(min(matrix.size())):
            matrix[i, i] += (2*self.n_factors + F.softplus(matrix[i, i]))
        # Make it a learnable parameter
        self.trial_peak_offset_covar_ltri = nn.Parameter(matrix)
        self.smoothness_budget = nn.Parameter(torch.zeros(self.n_factors-1, dtype=torch.float64))
        # solely to check if the covariance matrix is positive semi-definite
        # trial_peak_offset_covar_matrix = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        # bool((trial_peak_offset_covar_matrix == trial_peak_offset_covar_matrix.T).all() and (np.linalg.eigvals(trial_peak_offset_covar_matrix).real >= 0).all())
        # std_dev = np.sqrt(np.diag(trial_peak_offset_covar_matrix))
        # corr = np.diag(1/std_dev) @ trial_peak_offset_covar_matrix @ np.diag(1/std_dev)


    def init_ground_truth(self, beta=None, alpha=None, theta=None, pi=None,
                          config_peak_offsets=None, trial_peak_offset_covar_ltri=None):
        self.init_random()
        if beta is not None:
            self.beta = nn.Parameter(beta)
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
        if theta is not None:
            self.theta = nn.Parameter(theta)
        if pi is not None:
            nn_pi = pi.reshape(self.n_areas, -1)
            nn_pi = (nn_pi - nn_pi[:, 0])[:, 1:]
            self.pi = nn.Parameter(nn_pi)
        if config_peak_offsets is not None:
            self.config_peak_offsets = nn.Parameter(config_peak_offsets)
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
        beta[factor_indcs] = torch.tensor(torch.log(latent_factors[factor_indcs]))
        props = torch.sum(neuron_factor_access, dim=(0, 1)) / (K*C)
        # true_props = props/torch.sum(props)
        tr_props = torch.log(props) + torch.logsumexp(props, dim=0)
        tr_props = tr_props - tr_props[0]
        pi = tr_props[1:]
        self.init_ground_truth(n_factors, beta=beta, pi=pi)


    def sample_trial_offsets(self, n_configs, n_trials):
        n_factors = self.beta.shape[0]
        trial_peak_offset_samples = torch.randn(self.n_trial_samples, n_trials * n_configs, 2 * n_factors, dtype=torch.float64).view(
            self.n_trial_samples, n_trials, n_configs, 2 * n_factors)
        self.trial_peak_offsets = torch.einsum('lj,nrcj->nrcl', self.trial_peak_offset_covar_ltri, trial_peak_offset_samples)


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
        # self.trial_peak_offsets  N x R x C x 2AL
        # self.config_peak_offsets  # C x 2AL
        offsets = self.trial_peak_offsets + self.config_peak_offsets.unsqueeze(0).unsqueeze(1)
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


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors, n_areas):  # and first 2 entropy terms
        # output before summing: C x K x R x L x N
        # factors # L x T
        # warped_factors # L x T x N x R x C
        # neuron_factor_access  # K x L x C
        # Y # K x T x R x C
        neuron_factor_access = neuron_factor_access.permute(2, 0, 1)  # C x K x L
        warped_factors = warped_factors.permute(4,3,0,2,1)  # C x R x L x N x T
        factors = torch.exp(self.beta)  # exp(beta)
        log_warped_factors = torch.log(warped_factors)  # beta
        Y_sum_t = torch.sum(Y, dim=1).permute(2, 0, 1)  # C x K x R
        alpha = F.softplus(self.alpha)
        theta = F.softplus(self.theta)
        pi = F.softmax(torch.cat([torch.zeros(n_areas, 1), self.pi], dim=1), dim=1).flatten()
        log_alpha = torch.log(alpha)

        # U tensor items:
        # Y_times_beta  # C x K x R x L x T
        Y_times_beta = torch.einsum('ktrc,lt->ckrlt', Y, self.beta)
        # Y_times_beta  # C x K x L
        Y_times_beta = torch.sum(Y_times_beta, dim=(2, 4))
        # Y_sum_rt # C x K
        Y_sum_rt = torch.sum(Y_sum_t, dim=-1)
        # Y_sum_rt_plus_alpha  # C x K x L
        Y_sum_rt_plus_alpha = Y_sum_rt.unsqueeze(2) + alpha.unsqueeze(0).unsqueeze(1)
        # dt_exp_beta_plus_theta  # C x K x L
        dt_exp_beta_plus_theta = (self.dt * Y.shape[2] * torch.sum(factors, dim=-1) + theta).unsqueeze(0).unsqueeze(1)
        # log_dt_exp_beta_plus_theta  # C x K x L
        log_dt_exp_beta_plus_theta = torch.log(dt_exp_beta_plus_theta)
        # Y_sum_rt_plus_alpha_times_log_dt_exp_beta_plus_theta  # C x K x L
        Y_sum_rt_plus_alpha_times_log_dt_exp_beta_plus_theta = Y_sum_rt_plus_alpha * log_dt_exp_beta_plus_theta
        # Y_sum_rt_times_logalpha  # C x K x L
        Y_sum_rt_times_logalpha = torch.einsum('ck,l->ckl', Y_sum_rt, log_alpha)

        # V tensor items:
        # Y_times_warped_beta  # C x K x R x L x N x T
        Y_times_warped_beta = torch.einsum('ktrc,crlnt->ckrlnt', Y, log_warped_factors)
        # Y_times_warped_beta  # C x K x R x L x N
        Y_times_warped_beta = torch.sum(Y_times_warped_beta, dim=-1)
        # Y_sum_t_plus_alpha  # C x K x R x L
        Y_sum_t_plus_alpha = Y_sum_t.unsqueeze(3) + alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # dt_exp_warpedbeta_plus_theta  # C x R x L x N
        dt_exp_warpedbeta_plus_theta = self.dt * torch.sum(warped_factors, dim=-1) + theta.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        # log_dt_exp_warpedbeta_plus_theta  # C x R x L x N
        log_dt_exp_warpedbeta_plus_theta = torch.log(dt_exp_warpedbeta_plus_theta)
        # Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta  # C x K x R x L x N
        Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta = torch.einsum('ckrl,crln->ckrln', Y_sum_t_plus_alpha, log_dt_exp_warpedbeta_plus_theta)
        # Y_sum_t_times_logalpha  # C x K x R x L
        Y_sum_t_times_logalpha = torch.einsum('ckr,l->ckrl', Y_sum_t, log_alpha)

        # shared items:
        # log_theta  # L
        log_theta = torch.log(theta)
        # alpha_log_theta  # L
        alpha_log_theta = alpha * log_theta
        # log_pi  # L
        log_pi = torch.log(pi)
        L_a = factors.shape[0] // n_areas

        # U_tensor # C x K x L
        U_tensor = (Y_times_beta - Y_sum_rt_plus_alpha_times_log_dt_exp_beta_plus_theta +
                    alpha_log_theta.unsqueeze(0).unsqueeze(1) + Y_sum_rt_times_logalpha +
                    log_pi.unsqueeze(0).unsqueeze(1))
        # U_tensor # C x K x A x La (need to implement it this way so the softmax operation doesn't include the zero terms)
        U_tensor = U_tensor.reshape(*U_tensor.shape[:-1], n_areas, L_a)
        # W_CKL # C x K x L
        W_CKL = (neuron_factor_access * F.softmax(U_tensor, dim=-1).reshape(*U_tensor.shape[:-2], factors.shape[0])).detach()
        self.W_CKL = W_CKL  # for finding the posterior clustering probabilities

        # V_tensor # C x K x R x L x N
        V_tensor = (Y_times_warped_beta - Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta +
                    alpha_log_theta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4) +
                    Y_sum_t_times_logalpha.unsqueeze(4) +
                    log_pi.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
        # V_tensor # C x K x R x A x La x N  (need to implement it this way so the softmax and logsumexp operations doesn't include the zero terms)
        V_tensor = V_tensor.reshape(*V_tensor.shape[:3], n_areas, L_a, V_tensor.shape[-1])
        # W_CRN # C x K x R x A x N
        W_CRN = torch.logsumexp(V_tensor, dim=-2)
        # neuron_area_access  #  C x K x 1 x A x 1
        neuron_area_access = neuron_factor_access[:, :, [i * L_a for i in range(n_areas)]].unsqueeze(2).unsqueeze(4)
        # W_CRN # C x R x N
        W_CRN = (F.softmax(torch.sum(W_CRN * neuron_area_access, dim=(1, 3)), dim=-1)).detach()
        self.W_CRN = W_CRN  # for finding the posterior trial offsets

        # W_CKL # C x K x 1 x L x 1
        W_CKL = W_CKL.unsqueeze(2).unsqueeze(4)
        # W_CRN # C x 1 x R x 1 x N
        W_CRN = W_CRN.unsqueeze(1).unsqueeze(3)
        # W_tensor # C x K x R x L x N
        W_tensor = (W_CKL * W_CRN).detach()
        scale = 1/torch.prod(torch.tensor(W_tensor.shape[:3]))

        # a_KL  # C x K x L
        a_KL = (Y_sum_rt_plus_alpha/dt_exp_beta_plus_theta).detach()
        self.a_CKL = a_KL  # for finding the posterior neuron firing rates
        # b_KL  # C x K x L
        b_KL = (torch.digamma(Y_sum_rt_plus_alpha) - log_dt_exp_beta_plus_theta).detach()
        # Liklelihood Terms
        # a_KL_times_dt_exp_warpedbeta_plus_theta # C x K x R x L x N
        a_KL_times_dt_exp_warpedbeta_plus_theta = torch.einsum('ckl,crln->ckrln', a_KL, dt_exp_warpedbeta_plus_theta)
        # alpha_log_theta_plus_alpha_b_KL  # C x K x L
        alpha_log_theta_plus_b_KL = (alpha_log_theta.unsqueeze(0).unsqueeze(1) + torch.einsum('l,ckl->ckl', alpha, b_KL)).unsqueeze(2).unsqueeze(4)
        log_gamma_alpha = torch.lgamma(alpha).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)
        log_pi = log_pi.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)

        # trial_peak_offsets  # N x R x C x 2AL
        d = self.trial_peak_offset_covar_ltri.shape[0]
        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.t()
        det_Sigma = torch.linalg.det(Sigma)
        inv_Sigma = torch.linalg.inv(Sigma)
        prod_term = torch.einsum('nrcl,lj,nrcj->crn', self.trial_peak_offsets, inv_Sigma, self.trial_peak_offsets)  # sum over l
        normalizing_const = d*torch.log(torch.tensor(2*torch.pi)) + torch.log(det_Sigma)
        entropy_term = 0.5 * (normalizing_const + prod_term)
        entropy_term = entropy_term.unsqueeze(1).unsqueeze(3)

        elbo = Y_times_warped_beta - a_KL_times_dt_exp_warpedbeta_plus_theta - log_gamma_alpha + alpha_log_theta_plus_b_KL + log_pi - entropy_term
        elbo = scale * W_tensor * elbo

        return torch.sum(elbo)


    def compute_penalty_terms(self, tau_beta, tau_budget, tau_config, tau_sigma):
        # Penalty Terms
        config_Penalty = - tau_config * torch.sum(self.config_peak_offsets * self.config_peak_offsets)

        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty = -tau_sigma * (torch.sum(torch.abs(inv_Sigma)) - torch.sum(torch.abs(torch.diag(inv_Sigma))))

        latent_factors = torch.exp(self.beta)
        smoothness_budget_constrained = F.softmax(torch.cat([torch.zeros(1), self.smoothness_budget]), dim=0)
        beta_s2_penalty = - tau_beta * smoothness_budget_constrained.t() @ torch.sum((latent_factors @ self.Delta2TDelta2) * latent_factors, dim=1)

        smoothness_budget_penalty = - tau_budget * (self.smoothness_budget @ self.smoothness_budget)

        penalty_term = config_Penalty + sigma_Penalty + beta_s2_penalty + smoothness_budget_penalty
        return penalty_term


    def infer_latent_variables(self):
        # trial_offsets # C x R x 2AL
        trial_offsets = torch.einsum('nrcl,crn->crl', self.trial_peak_offsets, self.W_CRN)
        # likelihoods # C x K x L
        neuron_factor_assignment = torch.where(self.W_CKL == torch.max(self.W_CKL, dim=-1, keepdim=True).values, 1, 0)
        neuron_firing_rates = torch.sum(self.a_CKL * neuron_factor_assignment, dim=2)
        return trial_offsets.numpy(), neuron_factor_assignment.numpy(), neuron_firing_rates.numpy()


    def forward(self, Y, neuron_factor_access, n_areas, tau_beta, tau_budget, tau_config, tau_sigma):
        _, _, n_trials, n_configs = Y.shape
        self.sample_trial_offsets(n_configs, n_trials)
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors, n_areas)
        penalty_term = self.compute_penalty_terms(tau_beta, tau_budget, tau_config, tau_sigma)
        return likelihood_term, penalty_term


    def evaluate(self, Y, neuron_factor_access, n_areas):
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors, n_areas)
        trial_offsets, neuron_factor_assignment, neuron_firing_rates = self.infer_latent_variables()
        return likelihood_term, trial_offsets, neuron_factor_assignment, neuron_firing_rates
