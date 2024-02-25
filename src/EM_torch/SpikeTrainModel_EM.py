import numpy as np
import multiprocessing as mp
from src.EM_torch.model_data import ModelData
from src.psplines_gradient_method.general_functions import create_second_diff_matrix
from scipy.interpolate import BSpline
from scipy.sparse import csr_array, vstack, hstack
from scipy.special import psi, softmax, logsumexp, loggamma
from scipy.optimize import root

class SpikeTrainModel(ModelData):
    def __init__(self):
        super().__init__()

        # variables
        self.smoothness_budget = None  # L x 1

        # parameters
        self.BDelta2TDelta2BT = None  # T x T


    def initialize(self, Y, time, factor_access, n_factors, n_trial_samples, n_config_samples, degree=3):

        super().initialize(time, degree)

        # variables
        self.smoothness_budget = np.zeros((n_factors, 1))

        # parameters
        _, T, n_trials, n_configs = Y.shape
        self.randomly_initialize_parameters(n_factors, n_trials, n_configs, n_trial_samples, n_config_samples)
        self.Y = Y
        self.neuron_factor_access = factor_access
        Delta2BT = csr_array(create_second_diff_matrix(T)) @ self.B.T
        self.BDelta2TDelta2BT = Delta2BT.T @ Delta2BT

        return self

    def compute_log_likelihood(self, tau_beta=1, tau_sigma=1, tau_budget=1, derivs=False):
        # Weight Matrices
        (warped_factors,
         warped_time,
         sum_RT_Y_times_warped_bases,
         sum_RT_warped_factors_times_warped_bases,
         transformed_trial_peak_offsets,
         transformed_config_peak_offsets) = self.warp_all_latent_factors_for_all_trials(np.exp(self.beta), derivs)
        # warped_factors # L x T x M x N x R x C
        # sum_RT_Y_times_warped_bases # L x P x K x M x N x C
        # sum_RT_warped_factors_times_warped_bases # L x P x M x N x C
        # self.Y # K x T x R x C
        # Y_times_N_matrix  # K x L x T x M x N x R x C
        # sum_Y_times_N_matrix  # K x L x M x N x C
        # neuron_factor_access  #  C x K x L/A
        Y_times_N_matrix = np.einsum('ktrc,ltmnrc->kltmnrc', self.Y, warped_factors)
        sum_Y_times_N_matrix = np.sum(Y_times_N_matrix, axis=(2, 5))
        exp_N_matrix = np.exp(warped_factors)
        # sum_Y_term # K x C
        # logterm1  # K x C x L
        # logterm2  # L x M x N x C
        sum_Y_term = np.sum(self.Y, axis=(1,2)) # K x C
        logterm1 = sum_Y_term[:,:,None] + self.alpha[None,None,:]
        logterm2 = self.dt * np.sum(exp_N_matrix, axis=(1, 4)) + self.theta[:,None,None,None]
        # logterm # K x L x M x N x C
        logterm = np.einsum('kcl,lmnc->klmnc', logterm1, np.log(logterm2))
        # alphalogtheta # 1 x L x 1 x 1 x 1
        alphalogtheta = np.expand_dims(self.alpha * np.log(self.theta), (0,2,3,4))
        # sum_Y_times_logalpha  # K x C x L
        sum_Y_times_logalpha = sum_Y_term[:,:,None] * np.log(self.alpha)[None,None,:]
        # sum_Y_times_logalpha # K x L x 1 x 1 x C
        sum_Y_times_logalpha = np.transpose(sum_Y_times_logalpha, (0, 2, 1))[:,:,None,None,:]
        # logpi # 1 x L x 1 x 1 x 1
        logpi_expand = np.expand_dims(np.log(self.pi), (0,2,3,4))
        alpha_expand = np.expand_dims(self.alpha, (0,2,3,4))
        theta_expand = np.expand_dims(self.theta, (0,2,3,4))

        # U_tensor # K x L x M x N x C
        U_tensor = np.exp(sum_Y_times_N_matrix - logterm + alphalogtheta + sum_Y_times_logalpha + logpi_expand)

        # W_CMNK_tensor # K x L x M x N x C
        W_CMNK_tensor = softmax(U_tensor, axis=1)

        exp_sum_logsumexp_tensor = np.sum(logsumexp(U_tensor, axis=1), axis=0)
        exp_sum_logsumexp_tensor_reshape = exp_sum_logsumexp_tensor.reshape(-1, W_CMNK_tensor.shape[-1])

        # W_C_tensor # 1 x 1 x M x N x C
        W_C_tensor = softmax(exp_sum_logsumexp_tensor_reshape, axis=0).reshape(exp_sum_logsumexp_tensor.shape)[None,None,:,:,:]

        # W_tensor # K x L x M x N x C
        W_tensor = (W_CMNK_tensor * W_C_tensor)

        # A_tensor # K x L x M x N x C
        A_tensor = np.einsum('kcl,lmnc->klmnc', logterm1, 1 / logterm2)

        # Liklelihood Terms
        likelihood_term = (sum_Y_times_N_matrix - A_tensor * logterm2[None,:,:,:,:] - loggamma(alpha_expand) + alpha_expand *
                           (np.log(theta_expand) + np.transpose(psi(logterm1), (0, 2, 1))[:,:,None,None,:] - np.log(logterm2[None,:,:,:,:])) + logpi_expand)
        likelihood_term = np.sum(W_tensor * likelihood_term)

        # Entropy Terms
        dim = self.config_peak_offset_stdevs.shape[0]
        prod = np.einsum('ncl,l->ncl', transformed_config_peak_offsets, 1/self.config_peak_offset_stdevs)
        # entropy_term1  # N x C
        entropy_term1 = -0.5 * np.sum(np.log((2 * np.pi)**dim * np.prod(self.config_peak_offset_stdevs**2)) + prod**2, axis=2)
        # entropy_term1  # 1 x 1 x 1 x N x C
        entropy_term1 = np.sum(W_C_tensor * np.expand_dims(entropy_term1, (0,1,2)))

        Sigma = self.trial_peak_offset_covar_ltri @ self.trial_peak_offset_covar_ltri.T
        det_Sigma = np.linalg.det(Sigma)
        inv_Sigma = np.linalg.inv(Sigma)
        # entropy_term2  # M x C
        entropy_term2 = -0.5 * np.sum(np.log((2 * np.pi)**dim * det_Sigma) + np.einsum('mrci,ij,mrcj->mrc', transformed_trial_peak_offsets, inv_Sigma, transformed_trial_peak_offsets), axis=1)
        # entropy_term2  # 1 x 1 x M x 1 x C
        entropy_term2 = np.sum(W_C_tensor * np.expand_dims(entropy_term2, (0,1,3)))

        # penalty terms
        sigma_Penalty = - tau_sigma * np.sum(np.abs(inv_Sigma))

        latent_coefficients = np.exp(self.beta)
        smoothness_budget_constrained = softmax(self.smoothness_budget, axis=0)
        beta_s2_penalty = - tau_beta * smoothness_budget_constrained.T @ np.sum((latent_coefficients @ self.BDelta2TDelta2BT) * latent_coefficients, axis=1)

        smoothness_budget = - tau_budget * (self.smoothness_budget.T @ self.smoothness_budget)



        # beta_gradients = self.compute_beta_gradients_and_update_beta(sum_RT_warped_factors_times_warped_bases, sum_RT_Y_times_warped_bases, A_tensor, W_tensor, 0.01)


    def compute_beta_gradients_and_update_beta(self, sum_RT_warped_factors_times_warped_bases, sum_RT_Y_times_warped_bases, A_tensor, W_tensor, learning_rate, tau_beta):

        # sum_RT_Y_times_warped_bases # L x P x K x M x N x C
        # sum_RT_warped_factors_times_warped_bases # L x P x M x N x C
        # A_tensor # K x L x M x N x C
        # W_tensor # K x L x M x N x C
        sum_RT_warped_factors_times_warped_bases = np.transpose(sum_RT_warped_factors_times_warped_bases, (1,0,2,3,4))
        sum_RT_Y_times_warped_bases = np.transpose(sum_RT_Y_times_warped_bases, (2,1,0,3,4,5))
        # sum_RT_Y_times_warped_bases # K x P x L x M x N x C
        # sum_RT_warped_factors_times_warped_bases # P x L x M x N x C
        # inner_terms  # K x P x L x M x N x C
        inner_terms = sum_RT_Y_times_warped_bases - self.dt * np.einsum('klmnc,plmnc->kplmnc', A_tensor, sum_RT_warped_factors_times_warped_bases)
        weighted_inner_terms = np.einsum('klmnc,kplmnc->kplmnc', W_tensor, inner_terms)
        likelihood_gradient_component = np.exp(self.beta) * np.sum(weighted_inner_terms, axis=(0, 3, 4, 5)).T
        smoothness_budget_constrained = softmax(self.smoothness_budget, axis=0)
        penalty_gradient_component = - 2 * tau_beta * smoothness_budget_constrained * self.beta * (self.beta @ self.BDelta2TDelta2BT)

        beta_gradients = likelihood_gradient_component + penalty_gradient_component
        self.beta += learning_rate * beta_gradients

        return beta_gradients

    def compute_Sigma_gradients_and_update_Sigma(self, sum_RT_warped_factors_times_warped_bases, sum_RT_Y_times_warped_bases, A_tensor, W_tensor, learning_rate):
        # sum_RT_Y_times_warped_bases # L x P x K x M x N x C
        # sum_RT_warped_factors_times_warped_bases # L x P x M x N x C
        # A_tensor # K x L x M x N x C
        # W_tensor # K x L x M x N x C
        sum_RT_warped_factors_times_warped_bases = np.transpose(sum_RT_warped_factors_times_warped_bases,
                                                                (1, 0, 2, 3, 4))
        sum_RT_Y_times_warped_bases = np.transpose(sum_RT_Y_times_warped_bases, (2, 1, 0, 3, 4, 5))
        # sum_RT_Y_times_warped_bases # K x P x L x M x N x C
        # sum_RT_warped_factors_times_warped_bases # P x L x M x N x C
        # inner_terms  # K x P x L x M x N x C
        inner_terms = sum_RT_Y_times_warped_bases - self.dt * np.einsum('klmnc,plmnc->kplmnc', A_tensor,
                                                                        sum_RT_warped_factors_times_warped_bases)
        weighted_inner_terms = np.einsum('klmnc,kplmnc->kplmnc', W_tensor, inner_terms)
        beta_gradients = np.exp(self.beta) * np.sum(weighted_inner_terms, axis=(0, 3, 4, 5)).T

        self.beta += learning_rate * beta_gradients

        return beta_gradients






        # gradient of sigma_Penalty
        dlogL_dG_star = ((self.Y - lambda_del_t) @ betaStar_BPsi.T) * self.mask_G
        while ct < max_iters:
            G_star_minus = self.G_star + learning_rate * dlogL_dG_star
            G_star_plus = np.maximum(np.abs(G_star_minus) - tau_G * learning_rate, 0) * np.sign(G_star_minus)
            gen_grad_curr = (G_star_plus - self.G_star) / learning_rate
        sigma_Penalty_grad = - tau_sigma * torch.inverse(Sigma)

        # Total Loss
        total_loss = np.sum(W_tensor * likelihood_term + W_C_tensor * entropy_term1 + W_C_tensor * entropy_term2) + sigma_Penalty + beta_s2_penalty

        return total_loss

    def penalty_term(self, tau_beta, tau_s):


        smoothness_budget_penalty = tau_s * torch.sum(self.smoothness_budget ** 2)
        penalty = beta_s2_penalty + smoothness_budget_penalty

        # beta penalty gradients
        s2_component = - 2 * tau_beta *  smoothness_budget_constrained * latent_factors @ self.BDelta2TDelta2BT

        return -penalty, smoothness_budget_constrained



    def init_ground_truth(self, latent_factors, latent_coupling):
        V_inv = np.linalg.pinv(self.V.toarray().T)
        beta = latent_factors @ V_inv.T
        self.gamma = np.log(beta)
        self.c = np.zeros_like(self.c)
        self.chi = -1e10 * np.ones_like(self.chi)
        self.chi[latent_coupling == 1] = 0

    def trial_peaktime_gradient_for_timewarp(self, avg_peak_time, trial_peak_time):
        time = self.time
        warped_time_gradient = np.zeros_like(time)
        l = time[self.left_landmark]
        r = time[self.right_landmark]
        p = avg_peak_time
        s = trial_peak_time
        s_new = p + s
        if s_new < l:
            s_new = l
        elif s_new > r:
            s_new = r
        s_new = 0.45  # TODO: remove this
        for i in range(len(time)):
            t = time[i]
            if t < l:
                warped_time_gradient[i] = 0
            elif t < s_new:
                warped_time_gradient[i] = ((t-l)*(l-p))/(s_new-l)**2
            elif t < r:
                warped_time_gradient[i] = ((t-r)*(r-p))/(r-s_new)**2
            else:
                warped_time_gradient[i] = 0
        return warped_time_gradient

    def NDeriv_times_PhiDeriv_gradients(self):
        splines = self.B
        beta = self.beta[self.joint_factors_indices]
        factors = np.exp(beta) @ splines
        splines = BSpline(self.knots, beta.T, self.degree)
        splines_derivatives = splines.derivative()
        WarpedFactorsDerivs_x_WarpedTimesDerivs = []
        for i in range(factors.shape[0]):
            avg_peak_time = self.time[np.argmax(factors[i, self.left_landmark:self.right_landmark])]
            trial_factor_derivs = []
            for trial_peak_time in self.trial_peak_times:
                warped_time = self.warped_time(avg_peak_time, trial_peak_time)
                trial_deriv = splines_derivatives(warped_time)[np.newaxis,:,i] * self.trial_peaktime_gradient_for_timewarp(avg_peak_time, trial_peak_time)[np.newaxis,:]
                trial_factor_derivs.append(trial_deriv)
            WarpedFactorsDerivs_x_WarpedTimesDerivs.append(np.vstack(trial_factor_derivs))
        WarpedFactorsDerivs_x_WarpedTimesDerivs = np.stack(WarpedFactorsDerivs_x_WarpedTimesDerivs)
        return WarpedFactorsDerivs_x_WarpedTimesDerivs # L_joint x R x T

    def peaktime_gradients(self):

        # some of these matrices are three dimensional
        deltatExpN_times_WxA_matrix = self.dt * np.exp(self.trial_warped_factors) * self.WxA_matrix
        gradient_term = self.WxY_matrix[self.joint_factors_indices,:,:] - deltatExpN_times_WxA_matrix[self.joint_factors_indices,:,:]
        L, R, T = gradient_term.shape
        WarpedFactorsDerivs_x_WarpedTimesDerivs = self.NDeriv_times_PhiDeriv_gradients()
        sum_k_term = np.sum(gradient_term * WarpedFactorsDerivs_x_WarpedTimesDerivs, axis=(0, 2))[np.newaxis,:]
        sum_r_term = (1 / self.Sigma) * np.sum(self.trial_peak_times - self.C @ self.mu)
        S_gradients = sum_k_term - sum_r_term
        return S_gradients[0,:]


    def update_peaktimes(self, loss, factor=1e-2, alpha=0.1, max_iters=4):

        ct = 0
        learning_rate = 1
        S_gradients = self.peaktime_gradients()
        self.S_minus = np.copy(self.trial_peak_times)
        loss_next = loss
        while ct < max_iters:
            self.trial_peak_times = self.S_minus + learning_rate * S_gradients
            self.update_factor_terms()
            loss_next = self.compute_loss()
            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.sum(S_gradients * S_gradients):
                break
            learning_rate *= factor
            ct += 1
        if ct < max_iters:
            loss = loss_next
        else:
            ct = np.inf
            self.beta = self.beta_minus
        self.update_factor_terms()
        return loss, ct

    def alpha_gradient(self, alpha):
        (trial_warped_factors, YxN_matrix, alpha_pow_Y_theta_pow_alpha_pi_matrix, alpha_plus_Y_matrix,
         deltaSumN_plus_theta_matrix) = self.compute_posterior_terms()
        w_matrix = (np.exp(YxN_matrix) * alpha_pow_Y_theta_pow_alpha_pi_matrix) / deltaSumN_plus_theta_matrix ** alpha_plus_Y_matrix
        w_matrix = w_matrix / np.sum(w_matrix, axis=1)[:, np.newaxis]
        B_matrix = psi(alpha_plus_Y_matrix) - np.log(deltaSumN_plus_theta_matrix) # K x L
        alpha_gradient = (np.sum(w_matrix * B_matrix, axis=0))/np.sum(w_matrix, axis=0) + np.log(self.theta) - psi(self.alpha)
        return alpha_gradient

    def update_alpha(self):
        alpha = self.alpha
        result = root(self.alpha_gradient, alpha)
        if not result.success:
            raise ValueError("Root finding did not converge")
        return result.x

    def update_closed_form_params(self):
        (trial_warped_factors, YxN_matrix, alpha_pow_Y_theta_pow_alpha_pi_matrix, alpha_plus_Y_matrix,
         deltaSumN_plus_theta_matrix) = self.compute_posterior_terms()
        w_matrix = (np.exp(
            YxN_matrix) * alpha_pow_Y_theta_pow_alpha_pi_matrix) / deltaSumN_plus_theta_matrix ** alpha_plus_Y_matrix
        w_matrix = w_matrix / np.sum(w_matrix, axis=1)[:, np.newaxis]
        A_matrix = (alpha_plus_Y_matrix / deltaSumN_plus_theta_matrix)
        sum_k_w_matrix = np.sum(w_matrix, axis=0)
        # update theta
        self.theta = (sum_k_w_matrix @ self.alpha) / np.sum(w_matrix * A_matrix, axis=0)
        # update pi
        self.pi = sum_k_w_matrix / w_matrix.shape[0]
        # update mu
        self.mu = np.sum(self.trial_peak_times @ self.C) / np.sum(self.C, axis=0)
        # update Sigma
        self.Sigma = np.sum((self.trial_peak_times - self.C @ self.mu) ** 2) / self.C.shape[0]

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta, tau_s, beta_first=1,
                                                               time_warping=False,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.1, max_iters=4):
        # define parameters
        K, L = self.chi.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
        # exp_alpha_c = objects["exp_alpha_c"]
        # exp_zeta_c = objects["exp_zeta_c"]
        # kappa_norm = objects["kappa_norm"]
        # psi_norm = objects["psi_norm"]
        # time_matrix = objects["time_matrix"]
        B_sparse = objects["B_sparse"]
        psi_penalty = objects["psi_penalty"]
        kappa_penalty = objects["kappa_penalty"]
        beta_s2_penalty = objects["beta_s2_penalty"]
        d2_penalty = objects["d2_penalty"]
        s2_norm = objects["s2_norm"]
        maxes = objects["maxes"]
        sum_exps_chi = objects["sum_exps_chi"]
        sum_exps_chi_plus_gamma_B = objects["sum_exps_chi_plus_gamma_B"]
        max_gamma = objects["max_gamma"]
        beta_minus_max = objects["beta_minus_max"]
        loss = objects["loss"]
        log_likelihood_cache = objects["log_likelihood"]
        loss_0 = loss

        if beta_first:
            # smooth_gamma
            ct = 0
            learning_rate = gamma_factor
            exp_chi = np.vstack([np.exp(self.chi[k] + self.c[k] - maxes[k]) for k in range(K)])  # variable
            likelihood_component = exp_chi.T @ np.vstack(
                [(1 / (sum_exps_chi_plus_gamma_B[k]) * self.Y[k] - 1 / sum_exps_chi[k] * self.dt) @ b.transpose() for
                 k, b in enumerate(B_sparse)])
            s2_component = s2_norm * beta_minus_max @ self.BDelta2TDelta2BT
            dlogL_dgamma = beta_minus_max * np.exp(max_gamma) * (
                        likelihood_component - 2 * tau_beta * np.exp(max_gamma) * s2_component)
            while ct < max_iters:
                gamma_plus = self.gamma + learning_rate * dlogL_dgamma

                # set up variables to compute loss
                maxes = [np.max(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + gamma_plus) for k in range(K)]
                sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
                sum_exps_chi_plus_gamma_B = [
                    np.sum(np.exp(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + gamma_plus - maxes[k]),
                           axis=0)[np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
                log_likelihood = np.sum(
                    np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                               (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
                max_gamma = np.max(gamma_plus)
                beta_minus_max = np.exp(gamma_plus - max_gamma)
                beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (
                            s2_norm.T @ np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max,
                                               axis=1)).squeeze()
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s2_penalty + d2_penalty
                # Armijo condition, using Frobenius norm for matrices, but for maximization
                if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dgamma, ord='fro') ** 2:
                    break
                learning_rate *= gamma_factor
                ct += 1

            if ct < max_iters:
                ct_gamma = ct
                smooth_gamma = learning_rate
                loss = loss_next
                self.gamma = gamma_plus
                log_likelihood_cache = log_likelihood
            else:
                ct_gamma = np.inf
                smooth_gamma = 0
            loss_gamma = loss

            # set up variables to compute loss in next round
            maxes = [np.max(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + self.gamma) for k in range(K)]
            sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
            sum_exps_chi_plus_gamma_B = [
                np.sum(np.exp(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + self.gamma - maxes[k]), axis=0)[
                np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
            log_likelihood = np.sum(
                np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                           (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
            max_gamma = np.max(self.gamma)
            beta_minus_max = np.exp(self.gamma - max_gamma)

            # smooth_d2
            ct = 0
            learning_rate = 1
            diagBetaDeltaBeta = np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)[:, np.newaxis]
            dlogL_dd2 = tau_beta * s2_norm * (s2_norm.T - np.eye(L)) @ (
                        np.exp(2 * max_gamma) * diagBetaDeltaBeta + 2 * tau_s * self.d2)
            while ct < max_iters:
                d2_plus = self.d2 + learning_rate * dlogL_dd2

                # set up variables to compute loss
                d2_plus[0, :] = 0
                s2 = np.exp(d2_plus)
                s2_norm = (1 / np.sum(s2)) * s2
                beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (s2_norm.T @ diagBetaDeltaBeta).squeeze()
                d2_penalty = - tau_s * (d2_plus.T @ d2_plus).squeeze()
                # compute loss
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s2_penalty + d2_penalty

                # Armijo condition, using l2 norm, but for maximization
                if loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dd2 * dlogL_dd2):
                    break
                learning_rate *= d_factor
                ct += 1

            if ct < max_iters:
                ct_d2 = ct
                smooth_d2 = learning_rate
                loss = loss_next
                self.d2 = d2_plus
            else:
                ct_d2 = np.inf
                smooth_d2 = 0
            loss_d2 = loss

            # set up variables to compute loss in next round
            s2 = np.exp(self.d2)
            s2_norm = (1 / np.sum(s2)) * s2
            beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (s2_norm.T @ diagBetaDeltaBeta).squeeze()
            d2_penalty = - tau_s * (self.d2.T @ self.d2).squeeze()

            dlogL_dchi = 0
            ct_chi = 0
            smooth_chi = 0
            loss_chi = 0
            dlogL_dc = 0
            ct_c = 0
            smooth_c = 0
            loss_c = 0

        else:
            # smooth_chi
            ct = 0
            learning_rate = 1
            beta = np.exp(self.gamma)  # variable
            exp_chi = np.exp(self.chi)  # variable
            G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
            E = np.exp(self.c)  # variable
            E_beta_Bpsi = [E[k][:, np.newaxis] * beta @ b for k, b in enumerate(B_sparse)]
            GEBetaBPsi = [G[k][np.newaxis, :] @ e for k, e in enumerate(E_beta_Bpsi)]
            dlogL_dchi = G * np.vstack([np.sum(
                (1 / GEBetaBPsi[k] * E_beta_Bpsi[k] - 1) * self.Y[k] - (np.eye(L) - G[k]) @ E_beta_Bpsi[k] * self.dt,
                axis=1)
                                        for k in range(K)])
            while ct < max_iters:
                chi_plus = self.chi + learning_rate * dlogL_dchi

                # set up variables to compute loss
                chi_plus[:, 0] = 0
                exp_chi = np.exp(chi_plus)  # variable
                G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
                GEBetaBPsi = np.vstack([G[k] @ e for k, e in enumerate(E_beta_Bpsi)])
                # compute loss
                log_likelihood = np.sum(np.log(GEBetaBPsi) * self.Y - GEBetaBPsi * self.dt)
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s2_penalty + d2_penalty

                # Armijo condition, using Frobenius norm for matrices, but for maximization
                if (loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dchi, ord='fro') ** 2):
                    break
                learning_rate *= G_factor
                ct += 1

            if ct < max_iters:
                ct_chi = ct
                smooth_chi = learning_rate
                loss = loss_next
                self.chi = chi_plus
                log_likelihood_cache = log_likelihood
            else:
                ct_chi = np.inf
                smooth_chi = 0
            loss_chi = loss

            # smooth_c
            ct = 0
            learning_rate = 1
            betaB = [beta @ b for k, b in enumerate(B_sparse)]
            E_G = E * G
            dlogL_dc = E_G * np.vstack(
                [(1 / (E_G[k][np.newaxis, :] @ betaB[k]) * self.Y[k] - self.dt) @ betaB[k].T for k, b in
                 enumerate(B_sparse)])
            while ct < max_iters:
                c_plus = self.c + learning_rate * dlogL_dc

                # set up variables to compute loss
                maxes = [np.max(self.chi[k][:, np.newaxis] + c_plus[k][:, np.newaxis] + self.gamma) for k in range(K)]
                sum_exps_chi_plus_gamma_B = [
                    np.sum(np.exp(self.chi[k][:, np.newaxis] + c_plus[k][:, np.newaxis] + self.gamma - maxes[k]),
                           axis=0)[np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
                log_likelihood = np.sum(
                    np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                               (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s2_penalty + d2_penalty

                # Armijo condition, using Frobenius norm for matrices, but for maximization
                if (loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dc, ord='fro') ** 2):
                    break
                learning_rate *= G_factor
                ct += 1

            if ct < max_iters:
                ct_c = ct
                smooth_c = learning_rate
                loss = loss_next
                self.c = c_plus
                log_likelihood_cache = log_likelihood
            else:
                ct_c = np.inf
                smooth_c = 0
            loss_c = loss

            dlogL_dgamma = 0
            ct_gamma = 0
            smooth_gamma = 0
            loss_gamma = 0
            dlogL_dd2 = 0
            ct_d2 = 0
            smooth_d2 = 0
            loss_d2 = 0

        ################################################################################################################
        # if time_warping:
        #     # smooth_alpha
        #     ct = 0
        #     learning_rate = 1
        #     y_minus_lambda_del_t = self.Y - lambda_del_t
        #     knots_Bpsinminus1_1 = [
        #         (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
        #         time in time_matrix]
        #     knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
        #                            knots_Bpsinminus1_1]
        #     GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        #     GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        #     psi_norm_Omega = psi_norm @ self.Omega_psi_B
        #     dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack(
        #         [0.5 * max(self.time) * self.degree * np.sum(
        #             GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V] * R)), axis=1) -
        #          2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
        #          for q in range(Q)]).T
        #     # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        #     while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        #         alpha_plus = self.alpha + learning_rate * dlogL_dalpha
        #
        #         # set up variables to compute loss
        #         exp_alpha_c = (np.exp(alpha_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K,
        #                                                                                    axis=0)
        #         psi = exp_alpha_c @ self.U_ones  # variable
        #         psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        #         time_matrix = 0.5 * max(self.time) * np.hstack(
        #             [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #         B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
        #         GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #         lambda_del_t = GBetaBPsi * self.dt
        #         # compute loss
        #         log_likelihood = np.sum(np.log(GBetaBPsi) * self.Y - lambda_del_t)
        #         psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        #         loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty
        #
        #         # Armijo condition, using Frobenius norm for matrices, but for maximization
        #         if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dalpha, ord='fro') ** 2:
        #             break
        #         learning_rate *= alpha_factor
        #         ct += 1
        #
        #     if ct < max_iters:
        #         ct_alpha = ct
        #         smooth_alpha = learning_rate
        #         loss = loss_next
        #         self.alpha = alpha_plus
        #     else:
        #         ct_alpha = np.inf
        #         smooth_alpha = 0
        #     loss_alpha = loss
        #
        #     # set up variables to compute loss in next round
        #     exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K,
        #                                                                                axis=0)  # now fixed
        #     psi = exp_alpha_c @ self.U_ones  # now fixed
        #     psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        #     time_matrix = 0.5 * max(self.time) * np.hstack(
        #         [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #     B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in
        #                 time_matrix]  # variable
        #     GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #     lambda_del_t = GBetaBPsi * self.dt
        #     # compute updated penalty
        #     psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        #
        #     # smooth_zeta
        #     ct = 0
        #     learning_rate = 1
        #     y_minus_lambda_del_t = self.Y - lambda_del_t
        #     knots_Bpsinminus1_1 = [
        #         (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
        #         time in time_matrix]
        #     knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
        #                            knots_Bpsinminus1_1]
        #     GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        #     GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        #     kappa_norm_Omega = kappa_norm @ self.Omega_psi_B
        #     dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
        #                                                                          np.sum(np.sum(
        #                                                                              GBetaBPsiDerivXyLambda * ((
        #                                                                                                                self.U_ones[
        #                                                                                                                    q] - kappa_norm) @ self.V).flatten(),
        #                                                                              axis=0).reshape((R, T)), axis=1) -
        #                                                                          2 * tau_psi * np.sum(
        #         kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
        #                                                                          for q in range(Q)]).T
        #     # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        #     while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        #         zeta_plus = self.zeta + learning_rate * dlogL_dzeta
        #
        #         # set up variables to compute loss
        #         exp_zeta_c = (np.exp(zeta_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R,
        #                                                                                  axis=0)
        #         kappa = exp_zeta_c @ self.U_ones  # variable
        #         kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        #         time_matrix = 0.5 * max(self.time) * np.hstack(
        #             [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #         B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
        #         GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #         lambda_del_t = GBetaBPsi * self.dt
        #         # compute loss
        #         log_likelihood = np.sum(np.log(GBetaBPsi) * self.Y - lambda_del_t)
        #         kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        #         loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty
        #
        #         # Armijo condition, using Frobenius norm for matrices, but for maximization
        #         if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dzeta, ord='fro') ** 2:
        #             break
        #         learning_rate *= alpha_factor
        #         ct += 1
        #
        #     if ct < max_iters:
        #         ct_zeta = ct
        #         smooth_zeta = learning_rate
        #         loss = loss_next
        #         self.zeta = zeta_plus
        #     else:
        #         ct_zeta = np.inf
        #         smooth_zeta = 0
        #     loss_zeta = loss
        #
        #     # set up variables to compute loss in next round
        #     exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R,
        #                                                                              axis=0)  # now fixed
        #     kappa = exp_zeta_c @ self.U_ones  # now fixed
        #     kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # now fixed, called \kappa' in the document
        #     time_matrix = 0.5 * max(self.time) * np.hstack(
        #         [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # now fixed
        #     B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in
        #                 time_matrix]  # now fixed
        #     GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #     # compute updated penalty
        #     kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        #
        # else:
        dlogL_dalpha = 0
        smooth_alpha = 0
        ct_alpha = 0
        loss_alpha = 0
        dlogL_dzeta = 0
        smooth_zeta = 0
        ct_zeta = 0
        loss_zeta = 0

        result = {
            "dlogL_dgamma": dlogL_dgamma,
            "gamma_loss_increase": loss_gamma - loss_0,
            "smooth_gamma": smooth_gamma,
            "iters_gamma": ct_gamma,
            "dlogL_dd2": dlogL_dd2,
            "d2_loss_increase": loss_d2 - loss_gamma,
            "smooth_d2": smooth_d2,
            "iters_d2": ct_d2,
            "dlogL_dalpha": dlogL_dalpha,
            "alpha_loss_increase": loss_alpha - loss_d2,
            "smooth_alpha": smooth_alpha,
            "iters_alpha": ct_alpha,
            "dlogL_dzeta": dlogL_dzeta,
            "zeta_loss_increase": loss_zeta - loss_alpha,
            "smooth_zeta": smooth_zeta,
            "iters_zeta": ct_zeta,
            "dlogL_dchi": dlogL_dchi,
            "chi_loss_increase": loss_chi - loss_zeta,
            "smooth_chi": smooth_chi,
            "iters_chi": ct_chi,
            "dlogL_dc": dlogL_dc,
            "c_loss_increase": loss_c - loss_chi,
            "smooth_c": smooth_c,
            "iters_c": ct_c,
            "loss": loss,
            "log_likelihood": log_likelihood_cache,
            "beta_s2_penalty": beta_s2_penalty,
            "d2_penalty": d2_penalty,
            "psi_penalty": psi_penalty,
            "kappa_penalty": kappa_penalty
        }

        return result








        K, L = self.chi.shape
        R, Q = self.zeta.shape
        self.chi[:, 0] = 0
        self.d2[0, :] = 0
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
        psi = exp_alpha_c @ self.U_ones  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
        kappa = exp_zeta_c @ self.U_ones  # variable
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack(
            [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in
                    time_matrix]  # variable
        maxes = [np.max(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + self.gamma) for k in range(K)]
        sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
        sum_exps_chi_plus_gamma_B = [
            np.sum(np.exp(self.chi[k][:, np.newaxis] + self.c[k][:, np.newaxis] + self.gamma - maxes[k]), axis=0)[
            np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
        log_likelihood = np.sum(
            np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                       (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
        if time_warping:
            psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
            kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        else:
            psi_penalty = 0
            kappa_penalty = 0

        max_gamma = np.max(self.gamma)
        beta_minus_max = np.exp(self.gamma - max_gamma)
        s2 = np.exp(self.d2)
        s2_norm = (1 / np.sum(s2)) * s2
        beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (
                    s2_norm.T @ np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)).squeeze()
        # we can pretend np.exp(2 * max_gamma) is part of the penalty
        d2_penalty = - tau_s * (self.d2.T @ self.d2).squeeze()

        loss = log_likelihood + psi_penalty + kappa_penalty + beta_s2_penalty + d2_penalty

        result = {
            "B_sparse": B_sparse,
            "exp_alpha_c": exp_alpha_c,
            "exp_zeta_c": exp_zeta_c,
            "kappa_norm": kappa_norm,
            "psi_norm": psi_norm,
            "time_matrix": time_matrix,
            "loss": loss,
            "log_likelihood": log_likelihood,
            "psi_penalty": psi_penalty,
            "kappa_penalty": kappa_penalty,
            "beta_s2_penalty": beta_s2_penalty,
            "d2_penalty": d2_penalty,
            "s2_norm": s2_norm,
            "maxes": maxes,
            "sum_exps_chi": sum_exps_chi,
            "sum_exps_chi_plus_gamma_B": sum_exps_chi_plus_gamma_B,
            "max_gamma": max_gamma,
            "beta_minus_max": beta_minus_max
        }
        return result

    def compute_analytical_grad_time_warping(self, tau_psi, tau_beta, tau_s, time_warping=False):

        # define parameters
        K, L = self.chi.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
        B_sparse = objects["B_sparse"]
        s2_norm = objects["s2_norm"]
        maxes = objects["maxes"]
        sum_exps_chi = objects["sum_exps_chi"]
        sum_exps_chi_plus_gamma_B = objects["sum_exps_chi_plus_gamma_B"]
        max_gamma = objects["max_gamma"]
        beta_minus_max = objects["beta_minus_max"]

        # compute gradients
        exp_chi = np.vstack([np.exp(self.chi[k] + self.c[k] - maxes[k]) for k in range(K)])  # variable
        likelihood_component = exp_chi.T @ np.vstack(
            [(1 / (sum_exps_chi_plus_gamma_B[k]) * self.Y[k] - 1 / sum_exps_chi[k] * self.dt) @ b.transpose() for k, b
             in enumerate(B_sparse)])
        s2_component = s2_norm * beta_minus_max @ self.BDelta2TDelta2BT
        dlogL_dgamma = beta_minus_max * np.exp(max_gamma) * (
                    likelihood_component - 2 * tau_beta * np.exp(max_gamma) * s2_component)

        diagBetaDeltaBeta = np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)[:, np.newaxis]
        dlogL_dd2 = tau_beta * s2_norm * (s2_norm.T - np.eye(L)) @ (
                np.exp(2 * max_gamma) * diagBetaDeltaBeta + 2 * tau_s * self.d2)

        if time_warping:
            dlogL_dalpha = 0
            dlogL_dzeta = 0
            # y_minus_lambda_del_t = self.Y - lambda_del_t
            # knots_Bpsinminus1_1 = [
            #     (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
            #     time in time_matrix]
            # knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
            #                        knots_Bpsinminus1_1]
            # GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
            # GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
            # psi_norm_Omega = psi_norm @ self.Omega_psi_B
            # kappa_norm_Omega = kappa_norm @ self.Omega_psi_B
            # dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack(
            #     [0.5 * max(self.time) * self.degree * np.sum(
            #         GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V] * R)), axis=1) -
            #      2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
            #      for q in range(Q)]).T
            #
            # dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
            #                                                                      np.sum(np.sum(
            #                                                                          GBetaBPsiDerivXyLambda * ((
            #                                                                                                            self.U_ones[
            #                                                                                                                q] - kappa_norm) @ self.V).flatten(),
            #                                                                          axis=0).reshape((R, T)), axis=1) -
            #                                                                      2 * tau_psi * np.sum(
            #     kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
            #                                                                      for q in range(Q)]).T
        else:
            dlogL_dalpha = 0
            dlogL_dzeta = 0

        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        E = np.exp(self.c)  # variable
        E_beta_Bpsi = [E[k][:, np.newaxis] * beta @ b for k, b in enumerate(B_sparse)]
        GEBetaBPsi = [G[k][np.newaxis, :] @ e for k, e in enumerate(E_beta_Bpsi)]
        dlogL_dchi = G * np.vstack([np.sum(
            (1 / GEBetaBPsi[k] * E_beta_Bpsi[k] - 1) * self.Y[k] - (np.eye(L) - G[k]) @ E_beta_Bpsi[k] * self.dt,
            axis=1) for k in range(K)])

        betaB = [beta @ b for k, b in enumerate(B_sparse)]
        E_G = E * G
        dlogL_dc = E_G * np.vstack(
            [(1 / (E_G[k][np.newaxis, :] @ betaB[k]) * self.Y[k] - self.dt) @ betaB[k].T for k, b in
             enumerate(B_sparse)])

        return dlogL_dgamma, dlogL_dd2, dlogL_dalpha, dlogL_dzeta, dlogL_dchi, dlogL_dc

    def compute_grad_chunk(self, name, variable, i, eps, tau_psi, tau_beta, tau_s, loss, time_warping=False):

        J = variable.shape[1]
        grad_chunk = np.zeros(J)

        print(f'Starting {name} gradient chunk at row: {i}')

        for j in range(J):
            orig = variable[i, j]
            variable[i, j] = orig + eps
            objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
            loss_eps = objects['loss']
            grad_chunk[j] = (loss_eps - loss) / eps
            variable[i, j] = orig

        print(f'Completed {name} gradient chunk at row: {i}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta, tau_s, time_warping=False):

        eps = 1e-4
        # define parameters
        K, L = self.chi.shape
        R = self.trials

        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
        loss = objects['loss']
        pool = mp.Pool()

        # gamma gradient
        gamma_async_results = [
            pool.apply_async(self.compute_grad_chunk,
                             args=('gamma', self.gamma, l, eps, tau_psi, tau_beta, tau_s, loss))
            for l in range(L)]

        # d2 gradient
        d2_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('d2', self.d2, l, eps, tau_psi, tau_beta, tau_s, loss)) for
            l in range(L)]

        if time_warping:
            # zeta gradient
            zeta_async_results = [
                pool.apply_async(self.compute_grad_chunk,
                                 args=('zeta', self.zeta, r, eps, tau_psi, tau_beta, tau_s, loss)) for
                r in range(R)]

            # alpha gradient
            alpha_async_results = [
                pool.apply_async(self.compute_grad_chunk,
                                 args=('alpha', self.alpha, k, eps, tau_psi, tau_beta, tau_s, loss))
                for k
                in range(K)]

        # chi gradient
        chi_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('chi', self.chi, k, eps, tau_psi, tau_beta, tau_s, loss))
            for k in
            range(K)]

        # c gradient
        c_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('c', self.c, k, eps, tau_psi, tau_beta, tau_s, loss)) for k
            in
            range(K)]

        pool.close()
        pool.join()

        gamma_grad = np.vstack([r.get() for r in gamma_async_results])
        d2_grad = np.vstack([r.get() for r in d2_async_results])
        if time_warping:
            zeta_grad = np.vstack([r.get() for r in zeta_async_results])
            alpha_grad = np.vstack([r.get() for r in alpha_async_results])
        else:
            zeta_grad = 0
            alpha_grad = 0
        chi_grad = np.vstack([r.get() for r in chi_async_results])
        c_grad = np.vstack([r.get() for r in c_async_results])

        return gamma_grad, d2_grad, alpha_grad, zeta_grad, chi_grad, c_grad
