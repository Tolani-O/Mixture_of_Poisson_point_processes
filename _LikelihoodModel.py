import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Storage for use in the forward pass
        self.neuron_factor_access = None
        self.trial_peak_offsets = None  # CRN x 2AL
        self.W_CKL = None  # CKL

        # Parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri = None  # 2AL x 2AL


    def init_params(self, beta, alpha, theta, pi, config_peak_offset_stdevs, trial_peak_offset_covar_ltri,
                    n_configs, n_neurons):
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.pi = pi
        self.config_peak_offset_stdevs = config_peak_offset_stdevs
        self.trial_peak_offset_covar_ltri = trial_peak_offset_covar_ltri


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
        W_CKL = neuron_factor_access * F.softmax(U_tensor, dim=-1).reshape(*U_tensor.shape[:-2], factors.shape[0])
        self.W_CKL = W_CKL

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
        W_CRN = torch.sum(W_CRN * neuron_area_access, dim=(1, 3))

        # W_CKL # C x K x 1 x L x 1
        W_CKL = W_CKL.unsqueeze(2).unsqueeze(4)
        # W_CRN # C x 1 x R x 1 x N
        W_CRN = W_CRN.unsqueeze(1).unsqueeze(3)
        # W_tensor # C x K x R x L x N
        W_tensor = (W_CKL * W_CRN).detach()
        scale = 1/torch.prod(torch.tensor(W_tensor.shape[:3]))

        # a_KL  # C x K x L
        a_KL = Y_sum_rt_plus_alpha/dt_exp_beta_plus_theta
        # b_KL  # C x K x L
        b_KL = torch.digamma(Y_sum_rt_plus_alpha) - log_dt_exp_beta_plus_theta
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
        return entropy_term1, entropy_term2


    def compute_neuron_factor_assignment(self, Y, neuron_factor_access, warped_factors, n_areas):
        self.compute_log_elbo(Y, neuron_factor_access, warped_factors, n_areas)
        # likelihoods # K x L x C
        likelihoods = self.W_tensor.squeeze()
        neuron_factor_assignment = torch.where(likelihoods == torch.max(likelihoods, dim=1, keepdim=True).values, 1, 0)
        return neuron_factor_assignment


    def forward(self, Y, neuron_factor_access, n_areas):
        warped_factors = self.warp_all_latent_factors_for_all_trials()
        likelihood_term = (1 / (Y.shape[0] * Y.shape[-2] * Y.shape[-1])) * self.compute_log_elbo(Y, neuron_factor_access, warped_factors, n_areas)
        entropy_term1, entropy_term2 = self.compute_offset_entropy_terms()
        entropy_term = (1 / (Y.shape[-1])) * entropy_term1 + (1 / (Y.shape[-2] * Y.shape[-1])) * entropy_term2
        factor_assignment = self.compute_neuron_factor_assignment(Y, neuron_factor_access, warped_factors, n_areas)
        return likelihood_term, entropy_term, factor_assignment
