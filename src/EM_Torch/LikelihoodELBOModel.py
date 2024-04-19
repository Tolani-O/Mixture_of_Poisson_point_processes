import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM_Torch.general_functions import create_second_diff_matrix, create_first_diff_matrix
import numpy as np


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trial_samples):
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
        Delta2 = create_second_diff_matrix(T)
        # Delta2 = create_first_diff_matrix(T)
        self.Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2)  # T x T # tikhonov regularization

        # Storage for use in the forward pass
        self.neuron_factor_access = None
        self.trial_peak_offsets = None  # NRC x 2AL
        self.W_CKL = None  # C x K x L
        self.W_CRN = None  # C x R x N
        self.a_CKL = None  # C x K x L
        self.theta = torch.ones(self.n_factors, dtype=torch.float64)  # 1 x AL
        self.pi = F.softmax(torch.zeros(self.n_areas, self.n_factors // self.n_areas, dtype=torch.float64), dim=1).flatten()  # 1 x AL

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        # self.coupling = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri_diag = None
        self.trial_peak_offset_covar_ltri_offdiag = None
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
        self.standard_init()


    def standard_init(self):
        self.theta = torch.ones(self.n_factors, dtype=torch.float64)
        # self.coupling = nn.Parameter(torch.ones(self.n_factors, dtype=torch.float64))
        self.pi = F.softmax(torch.zeros(self.n_areas, self.n_factors // self.n_areas, dtype=torch.float64), dim=1).flatten()
        # self.smoothness_budget = nn.Parameter(torch.zeros(self.n_factors, dtype=torch.float64))


    def init_ground_truth(self, beta=None, alpha=None, theta=None, coupling=None, pi=None,
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
            self.theta = theta
        if pi is not None:
            self.pi = pi
        # if coupling is not None:
        #     self.coupling = nn.Parameter(coupling)
        if config_peak_offsets is not None:
            self.config_peak_offsets = nn.Parameter(config_peak_offsets)
        if trial_peak_offset_covar_ltri is not None:
            self.trial_peak_offset_covar_ltri_diag = nn.Parameter(trial_peak_offset_covar_ltri.diag())
            n_dims = 2 * self.n_factors
            indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
            self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(trial_peak_offset_covar_ltri[indices[0], indices[1]])


    def init_from_data(self, Y, factor_access, init='zeros'):
        Y, factor_access = Y.cpu(), factor_access.cpu()
        # Y # K x T x R x C
        # factor_access  # K x L x C
        K, T, R, C = Y.shape
        summed_neurons = torch.einsum('ktrc,klc->lt', Y, factor_access)
        latent_factors = summed_neurons + C * K * torch.rand(self.n_factors, T)
        # latent_factors[[a * L_a for a in range(self.n_areas)], :] = 1
        latent_factors = latent_factors / torch.sum(latent_factors, dim=-1, keepdim=True)
        beta = torch.log(latent_factors)
        spike_counts = torch.einsum('ktrc,klc->krlc', Y, factor_access)
        mean = torch.sum(spike_counts, dim=(0,1,3)) / (R * torch.sum(factor_access, dim=(0, 2)))
        alpha = mean

        # neurons_centered = (spike_counts - mean[None,None,:,None])**2
        # summed_neurons_centered = torch.einsum('krlc,klc->l', neurons_centered, factor_access)
        # varaince = summed_neurons_centered / (R * torch.sum(factor_access, dim=(0, 2)))
        # alpha = mean**2 / varaince
        # L_a = self.n_factors // self.n_areas
        # x_bar = torch.sum(summed_neurons, dim=-1)/(R*torch.sum(factor_access, dim=(0,2)))
        # log_x_bar = torch.log(x_bar)
        # log_x = torch.einsum('krc,klc->krlc', torch.log(torch.sum(Y,dim=1)), factor_access)
        # log_bar_x = torch.sum(log_x, dim=(0,1,3))/(R*torch.sum(factor_access, dim=(0,2)))
        # alpha = 0.5/(log_x_bar - log_bar_x)
        # alpha = torch.sum(summed_neurons, dim=-1)/(R*torch.sum(factor_access, dim=(0,2)))
        #latent_factors = np.apply_along_axis(gaussian_filter1d, axis=0, arr=summed_neurons, sigma=4).T
        # latent_factors[[a*L_a+1 for a in range(self.n_areas)], :] = (
        #         latent_factors[[a*L_a+1 for a in range(self.n_areas)], :] + np.random.uniform(low=0, high=K, size=(self.n_areas, T)))

        self.init_ground_truth(beta=beta, alpha=alpha, init=init)


    def cuda(self, device=None):
        self.device = 'cuda'
        self.time = self.time.cuda(device)
        self.theta = self.theta.cuda(device)
        self.pi = self.pi.cuda(device)
        self.Delta2TDelta2 = self.Delta2TDelta2.cuda(device)
        super(LikelihoodELBOModel, self).cuda(device)
        return self


    def cpu(self):
        self.device = 'cpu'
        self.time = self.time.cpu()
        self.theta = self.theta.cpu()
        self.pi = self.pi.cpu()
        self.Delta2TDelta2 = self.Delta2TDelta2.cpu()
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


    def sample_trial_offsets(self, n_configs, n_trials):
        n_factors = self.beta.shape[0]
        trial_peak_offset_samples = torch.randn(self.n_trial_samples, n_trials * n_configs, 2 * n_factors,
                                                device=self.device, dtype=torch.float64).view(
                                self.n_trial_samples, n_trials, n_configs, 2 * n_factors)
        ltri_matrix = self.ltri_matix()
        self.trial_peak_offsets = torch.einsum('lj,nrcj->nrcl', ltri_matrix, trial_peak_offset_samples)


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


    def compute_log_elbo(self, Y, neuron_factor_access, warped_factors):  # and first 2 entropy terms
        # output before summing: C x K x R x L x N
        # factors # L x T
        # warped_factors # L x T x N x R x C
        # neuron_factor_access  # K x L x C
        # Y # K x T x R x C
        K, T, R, C = Y.shape
        neuron_factor_access = neuron_factor_access.permute(2, 0, 1)  # C x K x L
        warped_factors = warped_factors.permute(4,3,0,2,1)  # C x R x L x N x T
        # include coupling
        # warped_factors = warped_factors**self.coupling.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        warped_factors = warped_factors / torch.sum(warped_factors, dim=-1, keepdim=True)
        log_warped_factors = torch.log(warped_factors)  # warped beta
        # include coupling
        # beta = self.coupling.unsqueeze(1) * self.beta
        factors = F.softmax(self.beta, dim=-1)  # exp(beta)
        beta = torch.log(factors)  # beta
        Y_sum_t = torch.sum(Y, dim=1).permute(2, 0, 1)  # C x K x R
        alpha = F.softplus(self.alpha)
        theta = self.theta
        pi = self.pi
        dt = 1

        # U tensor items:
        # Y_times_beta  # C x K x R x L x T
        Y_times_beta = torch.einsum('ktrc,lt->ckrlt', Y, beta)
        # Y_times_beta  # C x K x L
        Y_times_beta = torch.sum(Y_times_beta, dim=(2, 4))
        # Y_sum_rt # C x K
        Y_sum_rt = torch.sum(Y_sum_t, dim=-1)
        # Y_sum_rt_plus_alpha  # C x K x L
        Y_sum_rt_plus_alpha = Y_sum_rt.unsqueeze(2) + alpha.unsqueeze(0).unsqueeze(1)
        # dt_exp_beta_plus_theta  # C x K x L
        dt_exp_beta_plus_theta = (R * dt * torch.sum(factors, dim=-1) + theta).unsqueeze(0).unsqueeze(1)
        # log_dt_exp_beta_plus_theta  # C x K x L
        log_dt_exp_beta_plus_theta = torch.log(dt_exp_beta_plus_theta)
        # factors_times_1_minus_Y  # C x K x R x L x T
        factors_times_1_minus_Y = torch.einsum('ktrc,lt->ckrlt', 1 - Y, factors)
        # dt_factors_plus_theta  # C x K x L
        dt_factors_plus_theta = dt * torch.sum(factors_times_1_minus_Y, dim=(2, 4)) + theta.unsqueeze(0).unsqueeze(1)
        # log_dt_factors_plus_theta  # C x K x L
        log_dt_factors_plus_theta = torch.log(dt_factors_plus_theta)
        # Y_sum_rt_plus_alpha_times_log_dt_factors_plus_theta  # C x K x L
        Y_sum_rt_plus_alpha_times_log_dt_factors_plus_theta = Y_sum_rt_plus_alpha * log_dt_factors_plus_theta
        # Y_sum_rt_times_logalpha  # C x K x L
        Y_sum_rt_times_logalpha = torch.einsum('ck,l->ckl', Y_sum_rt, torch.log(alpha))

        # V tensor items:
        # Y_times_warped_beta  # C x K x R x L x N x T
        Y_times_warped_beta = torch.einsum('ktrc,crlnt->ckrlnt', Y, log_warped_factors)
        # Y_times_warped_beta  # C x K x R x L x N
        Y_times_warped_beta = torch.sum(Y_times_warped_beta, dim=-1)
        # Y_sum_t_plus_alpha  # C x K x R x L
        Y_sum_t_plus_alpha = Y_sum_t.unsqueeze(3) + alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # dt_exp_warpedbeta_plus_theta  # C x R x L x N
        dt_exp_warpedbeta_plus_theta = dt * torch.sum(warped_factors, dim=-1) + theta.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        # log_dt_exp_warpedbeta_plus_theta  # C x R x L x N
        log_dt_exp_warpedbeta_plus_theta = torch.log(dt_exp_warpedbeta_plus_theta)
        # Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta  # C x K x R x L x N
        Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta = torch.einsum('ckrl,crln->ckrln', Y_sum_t_plus_alpha, log_dt_exp_warpedbeta_plus_theta)
        # Y_sum_t_times_logalpha  # C x K x R x L
        Y_sum_t_times_logalpha = torch.einsum('ckr,l->ckrl', Y_sum_t, torch.log(alpha))

        # shared items:
        # alpha_log_theta  # L
        alpha_log_theta = alpha * torch.log(theta)
        # log_pi  # L
        log_pi = torch.log(pi)
        L_a = self.n_factors // self.n_areas

        # U_tensor # C x K x L
        U_tensor = (Y_times_beta - Y_sum_rt_plus_alpha_times_log_dt_factors_plus_theta +
                    alpha_log_theta.unsqueeze(0).unsqueeze(1) + Y_sum_rt_times_logalpha +
                    log_pi.unsqueeze(0).unsqueeze(1))
        # U_tensor # C x K x A x La
        U_tensor = U_tensor.reshape(*U_tensor.shape[:-1], self.n_areas, L_a)
        # W_CKL # C x K x L
        W_CKL = (neuron_factor_access * F.softmax(U_tensor, dim=-1).reshape(*U_tensor.shape[:-2], self.n_factors)).detach()
        self.W_CKL = W_CKL  # for finding the posterior clustering probabilities

        # V_tensor # C x K x R x L x N
        V_tensor = (Y_times_warped_beta - Y_sum_t_plus_alpha_times_log_dt_exp_warpedbeta_plus_theta +
                    alpha_log_theta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4) +
                    Y_sum_t_times_logalpha.unsqueeze(4) +
                    log_pi.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
        # V_tensor # C x K x R x A x La x N
        V_tensor = V_tensor.reshape(*V_tensor.shape[:3], self.n_areas, L_a, V_tensor.shape[-1])
        # W_CRN # C x K x R x A x N
        W_CRN = torch.logsumexp(V_tensor, dim=-2)
        # neuron_area_access  #  C x K x 1 x A x 1
        neuron_area_access = neuron_factor_access[:, :, [i * L_a for i in range(self.n_areas)]].unsqueeze(2).unsqueeze(4)
        # W_CRN # C x R x N
        W_CRN = (F.softmax(torch.sum(W_CRN * neuron_area_access, dim=(1, 3)), dim=-1)).detach()
        self.W_CRN = W_CRN  # for finding the posterior trial offsets

        # W_CKL # C x K x 1 x L x 1
        W_CKL = W_CKL.unsqueeze(2).unsqueeze(4)
        # W_CRN # C x 1 x R x 1 x N
        W_CRN = W_CRN.unsqueeze(1).unsqueeze(3)
        # W_tensor # C x K x R x L x N
        W_tensor = (W_CKL * W_CRN).detach()

        # neuron_factor_access  # C x K x 1 x L x 1
        neuron_factor_access = neuron_factor_access.unsqueeze(2).unsqueeze(4)
        # a_CKL  # C x K x 1 x L x 1
        a_CKL = (Y_sum_rt_plus_alpha/dt_exp_beta_plus_theta).unsqueeze(2).unsqueeze(4).detach()
        # b_CKL  # C x K x 1 x L x 1
        b_CKL = (torch.digamma(Y_sum_rt_plus_alpha) - log_dt_exp_beta_plus_theta).unsqueeze(2).unsqueeze(4).detach()
        alpha = alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4)
        # pad for numerical stability
        pad = 1 / torch.prod(torch.tensor(W_CKL.shape[:2]))
        W_CKL = W_CKL + pad * neuron_factor_access
        # W_CKL  # L
        W_L = torch.sum(W_CKL, dim=(0, 1), keepdim=True)
        # Wa_L  # L
        Wa_L = torch.sum(W_CKL * a_CKL, dim=(0, 1), keepdim=True)
        # update likelihood terms
        # pi = (W_L/torch.sum(neuron_factor_access, dim=(0, 1), keepdim=True).unsqueeze(2).unsqueeze(4)) + 1e-100
        # theta = (alpha * (W_L / (Wa_L + 1e-100))) + 1e-100
        pi = W_L / torch.sum(neuron_factor_access, dim=(0, 1), keepdim=True)
        theta = alpha * (W_L / Wa_L)

        self.a_CKL = a_CKL.squeeze()  # for finding the posterior neuron firing rates
        self.theta = theta.squeeze()  # for next iteration
        self.pi = pi.squeeze()  # for next iteration`

        # Liklelihood Terms
        # a_CKL_times_dt_exp_warpedbeta_plus_theta # C x K x R x L x N
        a_CKL_times_dt_exp_warpedbeta_plus_theta = a_CKL * (R * dt * torch.sum(warped_factors, dim=-1).unsqueeze(1) + theta)
        # alpha_log_theta_plus_alpha_b_KL  # C x K x R x L x N
        alpha_log_theta_plus_b_KL = alpha * (torch.log(theta) + b_CKL)
        log_gamma_alpha = torch.lgamma(alpha)
        # log_gamma_alpha = ((alpha-0.5)*torch.log(alpha) - alpha)
        log_pi = torch.log(pi)

        # trial_peak_offsets  # N x R x C x 2AL
        ltri_matrix = self.ltri_matix()
        entropy_term = self.Sigma_log_likelihood(self.trial_peak_offsets, ltri_matrix)

        elbo = Y_times_warped_beta + (1/R) * (-a_CKL_times_dt_exp_warpedbeta_plus_theta - log_gamma_alpha + alpha_log_theta_plus_b_KL + log_pi) + (1/K) * entropy_term
        elbo = W_tensor * elbo

        return torch.sum(elbo)

    def Sigma_log_likelihood(self, trial_peak_offsets, ltri_matrix):
        n_dims = ltri_matrix.shape[0]
        Sigma = ltri_matrix @ ltri_matrix.t()
        det_Sigma = torch.linalg.det(Sigma)
        inv_Sigma = torch.linalg.inv(Sigma)
        prod_term = torch.einsum('nrcl,lj,nrcj->crn', trial_peak_offsets, inv_Sigma, trial_peak_offsets)  # sum over l
        normalizing_const = n_dims * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_Sigma)
        entropy_term = - 0.5 * (normalizing_const + prod_term).unsqueeze(1).unsqueeze(3)
        return entropy_term

    def compute_penalty_terms(self, tau_beta, tau_config, tau_sigma):
        # Penalty Terms
        config_Penalty = - tau_config * torch.sum(self.config_peak_offsets * self.config_peak_offsets)

        ltri_matrix = self.ltri_matix()
        Sigma = ltri_matrix @ ltri_matrix.t()
        inv_Sigma = torch.linalg.inv(Sigma)
        sigma_Penalty = -tau_sigma * (torch.sum(torch.abs(inv_Sigma)) - torch.sum(torch.abs(torch.diag(inv_Sigma))))

        beta_s2_penalty = -tau_beta * torch.sum((self.beta @ self.Delta2TDelta2) * self.beta)


        # latent_factors = torch.exp(self.beta)
        # L, T = latent_factors.shape
        # smoothness_budget_constrained = torch.exp(self.smoothness_budget)
        # beta_s2_penalty = -tau_beta * torch.sum((latent_factors @ self.Delta2TDelta2.to(self.device)) * latent_factors)

        # beta_entropy_penalty = tau_beta_entropy * torch.sum(latent_factors * torch.log(latent_factors))

        # pi_collapse_penalty = - tau_budget * torch.sum(self.pi * self.pi)

        # factors_as_simplex = F.softmax(self.beta, dim=1)
        # beta_entropy_penalty = tau_beta_entropy * torch.sum(factors_as_simplex * torch.log(factors_as_simplex))
        # latent_factors = latent_factors - torch.mean(latent_factors, dim=1, keepdim=True) #+ 1e-10
        # beta_cor_penalty = latent_factors @ latent_factors.t()
        # # sd = beta_cor_penalty.diagonal().sqrt().reciprocal().diag()
        # # beta_cor_penalty = sd @ beta_cor_penalty @ sd
        # L_a = L // n_areas
        # mask = torch.zeros_like(beta_cor_penalty, device=self.device)
        # for i in range(n_areas):
        #     mask[i*L_a:(i+1)*L_a, i*L_a:(i+1)*L_a] = 1
        # # mask.diagonal().fill_(0)
        # beta_cor_penalty = -tau_cov * 0.5 * 1/T * torch.sum(beta_cor_penalty * mask)
        # beta_cor_penalty = 0 #-tau_cov * torch.sum(latent_factors * latent_factors)
        # smoothness_budget_penalty = 0 #- tau_budget * (self.smoothness_budget @ self.smoothness_budget)

        penalty_term = config_Penalty + sigma_Penalty + beta_s2_penalty
        # + beta_entropy_penalty + pi_collapse_penalty + beta_cor_penalty + smoothness_budget_penalty
        return penalty_term


    def infer_latent_variables(self):
        # trial_offsets # C x R x 2AL
        trial_offsets = torch.einsum('nrcl,crn->crl', self.trial_peak_offsets, self.W_CRN)
        # likelihoods # C x K x L
        neuron_factor_assignment = torch.where(self.W_CKL == torch.max(self.W_CKL, dim=-1, keepdim=True).values, 1, 0)
        neuron_firing_rates = torch.sum(self.a_CKL * neuron_factor_assignment, dim=2)
        return trial_offsets, neuron_factor_assignment, neuron_firing_rates


    def forward(self, Y, neuron_factor_access, tau_beta, tau_config, tau_sigma, trial_peak_offsets=None, sigma=0):
        if trial_peak_offsets is None:
            _, _, n_trials, n_configs = Y.shape
            self.sample_trial_offsets(n_configs, n_trials)
        else:
            self.trial_peak_offsets = trial_peak_offsets
        sigma = bool(sigma)
        self.beta.requires_grad = not sigma
        self.alpha.requires_grad = not sigma
        self.config_peak_offsets.requires_grad = sigma
        self.trial_peak_offset_covar_ltri_diag.requires_grad = sigma
        self.trial_peak_offset_covar_ltri_offdiag.requires_grad = sigma
        if sigma:
            warped_factors = self.warp_all_latent_factors_for_all_trials()
        else:
            _, _, n_trials, n_configs = Y.shape
            warped_factors = (torch.exp(self.beta).unsqueeze(2).unsqueeze(3).unsqueeze(4).
                              expand(-1, -1, self.n_trial_samples, n_trials, n_configs))
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
        penalty_term = self.compute_penalty_terms(tau_beta, tau_config, tau_sigma)
        return likelihood_term, penalty_term


    def evaluate(self, Y, neuron_factor_access, trial_peak_offsets=None):
        if trial_peak_offsets is None:
            _, _, n_trials, n_configs = Y.shape
            self.sample_trial_offsets(n_configs, n_trials)
        else:
            self.trial_peak_offsets = trial_peak_offsets
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = self.compute_log_elbo(Y, neuron_factor_access, warped_factors)
        trial_offsets, neuron_factor_assignment, neuron_firing_rates = self.infer_latent_variables()
        return likelihood_term, trial_offsets, neuron_factor_assignment, neuron_firing_rates
