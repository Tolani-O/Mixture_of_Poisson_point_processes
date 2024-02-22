import numpy as np
import multiprocessing as mp
from src.EM.model_data import ModelData
from src.psplines_gradient_method.general_functions import create_second_diff_matrix
from scipy.interpolate import BSpline
from scipy.sparse import csr_array, vstack, hstack
from scipy.special import psi, softmax
from scipy.optimize import root


class SpikeTrainModel(ModelData):
    def __init__(self, Y, time, trial_condition_design):
        super().__init__()
        # self.time = None
        # self.joint_factors_indices = None
        # self.degree = None
        # self.dt = None
        # self.knots = None
        # self.B = None
        # self.left_landmark1 = 20
        # self.mid_landmark1 = 45
        # self.right_landmark1 = 70
        # self.left_landmark2 = 120
        # self.mid_landmark2 = 145
        # self.right_landmark2 = 170
        # self.Y = None
        # self.trial_warped_factors = None
        # self.trial_warped_splines = None
        # self.trial_peak_offsets = None
        #
        # # parameters
        # self.beta = None  # AL x P
        # self.alpha = None  # 1 x AL
        # self.theta = None  # 1 x AL




        # variables

        self.Sigma = None  # 2AL x 2AL, we are DEFINITELY going to constrain this to be a sparse matrix
        self.trial_peak_times = None  # R x 2AL
        self.d2 = None  # 1 x L
        self.beta_minus = None  # L x P
        self.S_minus = None  # 1 x R

        # parameters
        self.Y = Y  # A x K x R x T
        self.time = time
        self.trials = self.Y.shape[1]
        self.dt = round(self.time[1] - self.time[0], 3)
        self.alpha_prime_multiply = None
        self.alpha_prime_add = None
        self.U_ones = None
        self.B = None
        self.knots = None
        self.BDelta1TDelta1BT = None
        self.Delta2BT = None
        self.Omega_psi_B = None
        self.degree = None
        self.left_landmark1 = None
        self.mid_landmark1 = None
        self.right_landmark1 = None
        self.left_landmark2 = None
        self.mid_landmark2 = None
        self.right_landmark2 = None
        self.joint_factors_indices = None  # 1 x A
        self.C = trial_condition_design # R x C
        self.w_matrix = None
        self.a_matrix = None
        self.b_matrix = None
        self.YxN_matrix = None
        self.deltatSumExpN_vector = None
        self.trial_warped_factors = None
        self.trial_warped_splines = None
        self.WxY_matrix = None
        self.WxA_matrix = None


    def initialize_for_time_warping(self, L, joint_factor_indices, degree=3):

        # parameters
        A, K, R, T = self.Y.shape
        P = T + 2
        Q = P  # will be equal to P now
        R, C = self.C.shape
        self.degree = 3
        self.knots = np.concatenate([np.repeat(self.time[0], degree), self.time, np.repeat(self.time[-1], degree)])
        self.knots[-1] = self.knots[-1] + self.dt
        self.left_landmark1 = 20
        self.mid_landmark1 = 45
        self.right_landmark1 = 80
        self.left_landmark2 = 120
        self.mid_landmark2 = 200
        self.right_landmark2 = 280
        self.joint_factors_indices = joint_factor_indices  # 1 x A


        # time warping b-spline matrix. Coefficients would be from psi
        self.B = BSpline.design_matrix(self.time, self.knots, self.degree).transpose()
        self.alpha_prime_multiply = np.eye(Q)
        self.alpha_prime_multiply[0, 0] = 0
        self.alpha_prime_multiply[1, 1] = 0
        self.alpha_prime_multiply = csr_array(self.alpha_prime_multiply)
        self.alpha_prime_add = np.zeros((1, Q))
        self.alpha_prime_add[:, 1] = 1
        self.U_ones = np.triu(np.ones((Q, Q)))
        Delta2BT = csr_array(create_second_diff_matrix(T)) @ self.B.T
        self.BDelta2TDelta2BT = Delta2BT.T @ Delta2BT
        self.Omega_psi_B = self.BDelta2TDelta2BT

        self.beta = np.random.rand(A*L, P)
        self.alpha = np.random.rand(A, L)
        self.theta = np.random.rand(A, L)
        self.pi = np.random.rand(A, L)
        self.trial_peak_times = np.random.rand(R, 2 * A * L)
        self.Sigma = np.random.rand(2*A*L, 2*A*L)
        self.mu = np.random.rand(2*A*L, C)
        self.compute_posterior_terms()

        # variables
        self.chi = np.random.rand(K, L)
        self.chi[:, 0] = 0
        self.c = np.random.rand(K, L)
        self.gamma = np.random.rand(L, P)
        self.gamma[0, :] = 0
        self.zeta = np.zeros((R, Q))
        self.d2 = np.zeros((L, 1))
        self.d2[0, :] = 0

        return self

    def init_ground_truth(self, latent_factors, latent_coupling):
        V_inv = np.linalg.pinv(self.V.toarray().T)
        beta = latent_factors @ V_inv.T
        self.gamma = np.log(beta)
        self.c = np.zeros_like(self.c)
        self.chi = -1e10 * np.ones_like(self.chi)
        self.chi[latent_coupling == 1] = 0

    def warped_time(self, avg_peak_times, trial_peak_times):
        time = self.time
        warped_time = np.zeros_like(time)
        l1 = time[self.left_landmark1]
        r1 = time[self.right_landmark1]
        p1 = avg_peak_times[0]
        s1 = trial_peak_times[0]
        s1_new = p1 + s1
        if s1_new < l1:
            s1_new = l1
        elif s1_new > r1:
            s1_new = r1
        l2 = time[self.left_landmark2]
        r2 = time[self.right_landmark2]
        p2 = avg_peak_times[1]
        s2 = trial_peak_times[1]
        s2_new = p2 + s2
        if s2_new < l2:
            s2_new = l2
        elif s2_new > r2:
            s2_new = r2
        for i in range(len(time)):
            t = time[i]
            if t < l:
                warped_time[i] = t
            elif t < s_new:
                warped_time[i] = (t-l)*(p-l)/(s_new-l)+l
            elif t < r:
                warped_time[i] = (t-s_new)*(r-p)/(r-s_new)+p
            else:
                warped_time[i] = t
        return warped_time

    def warped_latent_factors_trial(self, gamma, avg_peak_times, trial_peak_times): # warped factors for trial r
        warped_time = self.warped_time(avg_peak_times, trial_peak_times)
        warped_splines = BSpline.design_matrix(warped_time, self.knots, self.degree).transpose()
        warped_factor = gamma @ warped_splines
        return warped_splines, warped_factor

    def warped_latent_factors(self):
        splines = self.B
        factors = np.exp(self.beta) @ splines
        trial_warped_factors = []
        trial_warped_splines = []
        for l in range(factors.shape[0]):
            if l in self.joint_factors_indices:
                # cross trial average peak times
                avg_peak_time1 = self.time[np.argmax(factors[l, self.left_landmark1:self.right_landmark1])]
                avg_peak_time2 = self.time[np.argmax(factors[l, self.left_landmark2:self.right_landmark2])]
            else:
                avg_peak_time1 = self.time[self.mid_landmark1]
                avg_peak_time2 = self.time[self.mid_landmark2]
            trial_factors = []
            trial_splines = []
            for trial_peak_times in self.trial_peak_times:
                warped_splines, warped_factor = self.warped_latent_factors_trial(np.exp(self.beta[l,:]), [avg_peak_time1, avg_peak_time2], trial_peak_times[2*l:2*(l+1)])
                trial_factors.append(warped_factor)
                trial_splines.append(warped_splines)
            trial_warped_factors.append(np.vstack(trial_factors))
            trial_warped_splines.append(trial_splines)
        indices_to_keep = np.delete(np.arange(self.beta.shape[0]), self.joint_factors_indices)
        beta_complement = self.beta[indices_to_keep]
        factors = np.exp(beta_complement) @ splines
        for thisfac in factors:
            trial_warped_factors.append(np.vstack([thisfac[np.newaxis,:]] * self.trial_peak_times.shape[0]))
            trial_warped_splines.append([splines] * self.trial_peak_times.shape[0])
        trial_warped_factors = np.stack(trial_warped_factors)
        return trial_warped_splines, trial_warped_factors # L x R x P x T, L x R x T

    def compute_posterior_terms(self):
        trial_warped_splines, trial_warped_factors = self.warped_latent_factors()
        # Compute YxN_matrix for all k, l in one go
        YxN_matrix = np.einsum('krt,lrt->kl', self.Y, trial_warped_factors).T
        # Compute alpha_pow_Y_theta_pow_alpha_pi_matrix for all k, l in one go
        alpha = self.alpha.reshape((-1, 1))
        # Compute Y_k_sum for all k in  one go
        Y_k_sum = np.sum(self.Y, axis=(1, 2))
        Y_k_sum_reshape = Y_k_sum.reshape((1, -1))
        # alpha_pow_Y_theta_pow_alpha_pi_matrix = (np.power(alpha, Y_k_sum_reshape) *
        #                                          np.power(self.theta, self.alpha)[:, np.newaxis] *
        #                                          self.pi[:, np.newaxis])
        # Compute alpha_plus_Y_matrix for all k, l in one go
        alpha_plus_Y_matrix = self.alpha[:, np.newaxis] + Y_k_sum
        # Compute deltatSumExpN_vector for all k, l in one go
        deltatSumExpN_vector = self.dt * np.sum(np.exp(trial_warped_factors), axis=(1, 2)) + self.theta
        deltatSumExpN_plus_theta_vector = deltatSumExpN_vector + self.theta

        w_matrix = (YxN_matrix - alpha_plus_Y_matrix * np.log(deltatSumExpN_plus_theta_vector[:, np.newaxis]) +
            self.alpha[:, np.newaxis] * np.log(self.theta[:, np.newaxis]) + np.dot(alpha, Y_k_sum_reshape) +
            np.log(self.pi[:, np.newaxis])) # K x L
        w_matrix = softmax(w_matrix, axis=0)
        # w_matrix2 = (np.exp(YxN_matrix) * alpha_pow_Y_theta_pow_alpha_pi_matrix) / deltatSumExpN_plus_theta_vector[:, np.newaxis] ** alpha_plus_Y_matrix
        # w_matrix2 = w_matrix / np.sum(w_matrix, axis=1)[:, np.newaxis]
        A_matrix = (alpha_plus_Y_matrix / deltatSumExpN_plus_theta_vector[:, np.newaxis])
        B_matrix = psi(alpha_plus_Y_matrix) - np.log(deltatSumExpN_plus_theta_vector[:, np.newaxis])  # K x L
        self.w_matrix = w_matrix
        self.a_matrix = A_matrix
        self.b_matrix = B_matrix
        self.YxN_matrix = YxN_matrix
        self.deltatSumExpN_vector = deltatSumExpN_vector
        self.trial_warped_factors = trial_warped_factors
        self.trial_warped_splines = trial_warped_splines
        self.WxY_matrix = np.einsum('lk,krt->lrt', self.w_matrix, self.Y)
        self.WxA_matrix = np.sum(self.w_matrix * self.a_matrix, axis=1)[:, np.newaxis, np.newaxis]

    def compute_log_likelihood(self):

        LK = self.w_matrix * (self.YxN_matrix - self.a_matrix*(self.deltatSumExpN_vector[:, np.newaxis] + self.theta[:, np.newaxis]) -
                              np.log(self.alpha[:, np.newaxis]) + self.alpha[:, np.newaxis]*(np.log(self.theta[:, np.newaxis]) + self.b_matrix) +
                              np.log(self.pi[:, np.newaxis]))
        LR = -(1/2) * np.log(2 * np.pi * self.Sigma) - (1 / 2) * ((self.trial_peak_times - self.C @ self.mu) / np.sqrt(self.Sigma)) ** 2
        L = np.sum(LK) + np.sum(LR)
        return L

    def beta_gradients(self):

        # some of these matrices are three dimensional
        deltatExpN_times_WxA_matrix = self.dt * np.exp(self.trial_warped_factors) * self.WxA_matrix
        gradient_term = self.WxY_matrix - deltatExpN_times_WxA_matrix
        L, R, T = gradient_term.shape
        P = self.beta.shape[1]
        beta_gradients = np.zeros((L, P))
        for l in range(L):
            factor_gradient_intermediate = np.zeros((P, R))
            for r in range(R):
                factor_gradient_intermediate[:,r,np.newaxis] = self.trial_warped_splines[l][r] @ gradient_term[l, r, :, np.newaxis]
            beta_gradients[l,:] = np.sum(factor_gradient_intermediate, axis=1)
        beta_gradients = np.exp(self.beta) * beta_gradients
        return beta_gradients

    def update_factor_terms(self):
        trial_warped_splines, trial_warped_factors = self.warped_latent_factors()
        # Compute YxN_matrix for all k, l in one go
        self.YxN_matrix = np.einsum('krt,lrt->kl', self.Y, trial_warped_factors).T
        self.deltatSumExpN_vector = self.dt * np.sum(np.exp(trial_warped_factors), axis=(1, 2))
        self.trial_warped_factors = trial_warped_factors
        self.trial_warped_splines = trial_warped_splines

    def update_beta(self, loss, factor=1e-2, alpha=0.1, max_iters=4):

        ct = 0
        learning_rate = 1
        beta_gradients = self.beta_gradients()
        self.beta_minus = np.copy(self.beta)
        loss_next = loss
        while ct < max_iters:
            self.beta = self.beta_minus + learning_rate * beta_gradients
            self.update_factor_terms()
            loss_next = self.compute_loss()
            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.linalg.norm(beta_gradients, ord='fro') ** 2:
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
