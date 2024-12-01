import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
torch.set_default_dtype(torch.float64)

def create_precision_matrix(P):
    Omega = np.zeros((P, P))
    # fill the main diagonal with 2s
    np.fill_diagonal(Omega, 2)
    # fill the subdiagonal and superdiagonal with -1s
    np.fill_diagonal(Omega[1:], -1)
    np.fill_diagonal(Omega[:, 1:], -1)
    # set the last element to 1
    Omega[-1, -1] = 1
    return Omega
def create_first_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 2:], 1)
    # first row is a forward difference
    s0 = [-1, 1]
    D0 = np.concatenate((s0, np.zeros(P - len(s0))))
    D = P * np.vstack((D0, D/2, -np.flip(D0)))
    return D
def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # first row is a forward difference
    s0 = [1, -2, 1]
    D0 = np.concatenate((s0, np.zeros(P-len(s0))))
    D = P**2 * np.vstack((D0, D, np.flip(D0)))
    return D


class LikelihoodELBOModel(nn.Module):
    def __init__(self, time, n_factors, n_areas, n_configs, n_trials, n_trial_samples,
                 peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                 temperature=None, weights=None, adjust_landmarks=False):
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
        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.time = time
        self.dt = torch.round(time[1] - time[0], decimals=3)
        T = time.shape[0]
        peak1_left_landmarks_indx = torch.searchsorted(self.time, peak1_left_landmarks, side='left')
        peak1_right_landmarks_indx = torch.searchsorted(self.time, peak1_right_landmarks, side='left')
        peak2_left_landmarks_indx = torch.searchsorted(self.time, peak2_left_landmarks, side='left')
        peak2_right_landmarks_indx = torch.searchsorted(self.time, peak2_right_landmarks, side='left')
        self.left_landmarks_indx = torch.cat([peak1_left_landmarks_indx, peak2_left_landmarks_indx])
        self.right_landmarks_indx = torch.cat([peak1_right_landmarks_indx, peak2_right_landmarks_indx])
        self.landmark_indx_speads = self.right_landmarks_indx - self.left_landmarks_indx + 1 # adding one because the last element is not included when indexing
        self.half_warping_window = 0.3 * (self.time[self.right_landmarks_indx] - self.time[self.left_landmarks_indx]) / 4  # 0.3 is the standard deviation for the gaussian distribution such that 99.7% of the values are in the range [-1, 1]
        self.n_factors = n_factors
        self.n_areas = n_areas
        self.n_trial_samples = n_trial_samples
        self.n_configs = n_configs
        self.n_trials = n_trials
        self.adjust_landmarks = adjust_landmarks
        self.Delta2 = torch.tensor(create_second_diff_matrix(T))
        self.Delta2TDelta2 = self.Delta2.T @ self.Delta2  # T x T # tikhonov regularization

        # Storage for use in the forward pass
        self.trial_peak_offset_proposal_samples = None  # N x R x C x 2AL
        # There are not learned but computed in the forward pass
        self.theta = None  # 1 x AL
        self.pi = None  # 1 x AL
        self.W_CKL = None  # C x K x L
        self.a_CKL = None  # C x K x L
        self.log_sum_exp_U_tensor = None  # C x K x L
        self.prec_ltri = None
        self.sigma_ltri = None

        # Parameters
        self.beta = None  # AL x T
        self.alpha = None  # 1 x AL
        self.config_peak_offsets = None  # C x 2AL
        self.trial_peak_offset_covar_ltri_diag = None
        self.trial_peak_offset_covar_ltri_offdiag = None
        self.trial_peak_offset_proposal_means = None  # R x C x 2AL
        self.trial_peak_offset_proposal_sds = None  # R x C x 2AL


    def init_random(self):
        self.beta = nn.Parameter(torch.log(torch.rand(self.n_factors, self.time.shape[0]-1)))
        self.alpha = nn.Parameter(torch.rand(self.n_factors))
        n_dims = 2 * self.n_factors
        self.config_peak_offsets = nn.Parameter(torch.randn(self.n_configs, n_dims))
        self.init_sigma_ltri(rand=True)
        self.trial_peak_offset_proposal_means = nn.Parameter(self.init_trial_offsets())
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.rand(self.n_trials, self.n_configs, n_dims) + 1)
        self.pi = F.softmax(torch.randn(self.n_areas, self.n_factors // self.n_areas), dim=1).flatten()
        self.theta = torch.rand(self.n_factors)
        self.standard_init()


    def init_zero(self):
        self.beta = nn.Parameter(torch.zeros(self.n_factors, self.time.shape[0]-1))
        self.alpha = nn.Parameter(torch.ones(self.n_factors))
        n_dims = 2 * self.n_factors
        self.config_peak_offsets = nn.Parameter(torch.zeros(self.n_configs, n_dims))
        self.init_sigma_ltri()
        self.trial_peak_offset_proposal_means = nn.Parameter(self.init_trial_offsets())
        self.trial_peak_offset_proposal_sds = nn.Parameter(torch.ones(self.n_trials, self.n_configs, n_dims))
        self.pi = F.softmax(torch.zeros(self.n_areas, self.n_factors // self.n_areas), dim=1).flatten()
        self.theta = torch.ones(self.n_factors)
        self.standard_init()


    def init_sigma_ltri(self, ltri_matrix=None, rand=False):
        n_dims = 2 * self.n_factors
        if ltri_matrix is None:
            if rand:
                ltri_matrix = torch.tril(torch.randn(n_dims, n_dims))
            else:
                ltri_matrix = torch.eye(n_dims, n_dims)
        sigma = ltri_matrix @ ltri_matrix.t()
        corr = sigma / torch.outer(sigma.diag(), sigma.diag()).sqrt()
        scaled_sigma = (corr * torch.outer(self.half_warping_window, self.half_warping_window)) + (torch.eye(n_dims, n_dims) * 1e-6)
        sigma_ltri = torch.linalg.cholesky(scaled_sigma)
        self.trial_peak_offset_covar_ltri_diag = nn.Parameter(sigma_ltri.diag())
        indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
        self.trial_peak_offset_covar_ltri_offdiag = nn.Parameter(sigma_ltri[indices[0], indices[1]])

    def init_trial_offsets(self):
        self.ltri_matix()
        normal_samples = torch.randn((self.n_trials, self.n_configs, 2 * self.n_factors))
        return torch.einsum('lj,rcj->rcl', self.sigma_ltri, normal_samples)


    def standard_init(self):
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
            beta = (beta - beta[:, 0].unsqueeze(1).expand_as(beta))[:, 1:]
            self.beta = nn.Parameter(beta)
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
        if theta is not None:
            self.theta = theta
        if W_CKL is not None:
            self.validated_W_CKL(W_CKL)
        elif pi is not None:
            self.pi = pi
        if sd_init is not None:
            self.trial_peak_offset_proposal_sds = nn.Parameter(sd_init * torch.ones(self.n_trials, self.n_configs, n_dims, device=self.device))
        if config_peak_offsets is not None:
            self.config_peak_offsets = nn.Parameter(config_peak_offsets)
        if trial_peak_offset_covar_ltri is not None:
            self.init_sigma_ltri(trial_peak_offset_covar_ltri)
            self.trial_peak_offset_proposal_means = nn.Parameter(self.init_trial_offsets())
        if trial_peak_offset_proposal_means is not None:
            self.trial_peak_offset_proposal_means = nn.Parameter(trial_peak_offset_proposal_means)


    def init_from_data(self, Y, factor_access, sd_init, cluster_dir=None, init='zeros'):
        # Y # K x T x R x C
        # factor_access  # C x K x L
        _, T, R, _ = Y.shape
        if cluster_dir is None:
            W_CKL = None
            summed_neurons = torch.einsum('ktrc,ckl->lt', Y, factor_access)
            latent_factors = summed_neurons + torch.sqrt(torch.sum(summed_neurons, dim=-1)).unsqueeze(1) * torch.rand(self.n_factors, T)
            beta = torch.log(latent_factors)
            filter = factor_access
        else:
            cluster_dir = os.path.join(cluster_dir, 'cluster_initialization.pkl')
            if not os.path.exists(cluster_dir):
                raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
            print('Loading clusters from: ', cluster_dir)
            with open(cluster_dir, 'rb') as f:
                data = pickle.load(f)
            W_CKL, beta = data['neuron_factor_assignment'], data['beta']
            filter = W_CKL
        # NOTE: Empirical average and variance of spike counts are for NB,not for gamma
        spike_counts = torch.einsum('ktrc,ckl->ckl', Y, filter)
        avg_spike_counts = torch.sum(spike_counts, dim=(0, 1)) / torch.sum(filter, dim=(0, 1))
        print('Gamma average spike counts:')
        print(avg_spike_counts.reshape(self.n_areas, -1).numpy() / R)
        sq_centered_spike_counts = (spike_counts - avg_spike_counts.unsqueeze(0).unsqueeze(1))**2 * filter
        spike_ct_var = torch.sum(sq_centered_spike_counts, dim=(0,1)) / (torch.sum(filter, dim=(0, 1)))
        dispersion = spike_ct_var-avg_spike_counts
        print('Gamma spike count sd:')
        print(torch.sqrt(dispersion).reshape(self.n_areas, -1).numpy() / R)
        alpha = (avg_spike_counts)**2/dispersion
        alpha = alpha.expm1().clamp_min(1e-6).log()
        theta = R * avg_spike_counts/dispersion
        self.init_ground_truth(beta=beta, alpha=alpha, theta=theta, sd_init=sd_init, W_CKL=W_CKL, init=init)
        Y_sum_rt_plus_alpha = Y.sum(dim=(1,2)).t().unsqueeze(-1) + alpha.unsqueeze(0).unsqueeze(1)  # C x K x L
        self.update_params(Y_sum_rt_plus_alpha, factor_access, R)

    def find_factor_peaks(self, top_k=2, half_window=4, sigma=2):
        factors = self.unnormalized_log_factors().exp()
        hess = -(self.Delta2 @ factors.t()).t()
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2).to(device=self.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, -1)  # Reshape for convolution
        # Apply convolution
        padding = (kernel.size(-1) // 2,)
        second_derivative = F.conv1d(hess.unsqueeze(1), kernel, padding=padding, groups=1).squeeze(1)
        # Identify peaks
        peaks = (second_derivative[:, 1:-1] > second_derivative[:, :-2]) & (second_derivative[:, 1:-1] > second_derivative[:, 2:])
        peak_indices = [torch.nonzero(row).squeeze() + 1 for row in peaks]
        # Collect top K peaks for each row
        top_peak_indices = []
        top_peak_values = []
        top_peak_times = []
        for i, indices in enumerate(peak_indices):
            if indices.numel() > 0:  # Ensure there are peaks
                # Get values of the peaks
                peak_values = second_derivative[i, indices]
                # Sort by values and select top K
                sorted_indices = torch.argsort(peak_values, descending=True)[:top_k]
                top_indices = indices[sorted_indices].sort().values
                final_indices = []
                for j in top_indices:
                    search_span = np.arange(j-half_window, j+half_window)
                    final_indices.append(search_span[torch.argmax(factors[i, search_span])])
                top_peak_indices.append(torch.tensor(final_indices))
                top_peak_values.append(factors[i, final_indices])
                top_peak_times.append(self.time[final_indices])
            else:
                # No peaks found
                top_peak_indices.append(torch.tensor([]))
                top_peak_values.append(torch.tensor([]))
                top_peak_times.append(torch.tensor([]))
        return torch.stack(top_peak_indices), torch.stack(top_peak_values), torch.stack(top_peak_times)


    def update(self):
        if self.adjust_landmarks:
            with torch.no_grad():
                top_peak_indices, top_peak_values, top_peak_times = self.find_factor_peaks()
                steps = self.left_landmarks_indx[self.n_factors:] - self.right_landmarks_indx[:self.n_factors]
                peal1_right_landmark_indx = torch.tensor([top_peak_indices[l, 0] + self.beta[l, top_peak_indices[l, 0]:top_peak_indices[l, 1]].argmin().item() for l in range(self.n_factors)])
                peak2_left_landmark_indx = peal1_right_landmark_indx + steps
                self.right_landmarks_indx[:self.n_factors] = peal1_right_landmark_indx
                self.left_landmarks_indx[self.n_factors:] = peak2_left_landmark_indx
                self.landmark_indx_speads = self.right_landmarks_indx - self.left_landmarks_indx + 1
                self.half_warping_window = 0.3 * (self.time[self.right_landmarks_indx] - self.time[self.left_landmarks_indx]) / 4


    # move to cuda flag tells the function whether gpus are available
    def cuda(self, device=None, move_to_cuda=True):
        if (not move_to_cuda) or (self.device == 'cuda'):
            return
        self.device = 'cuda'
        self.time = self.time.cuda(device)
        self.temperature = self.temperature.cuda(device)
        self.weights = self.weights.cuda(device)
        self.Delta2TDelta2 = self.Delta2TDelta2.cuda(device)
        self.Delta2 = self.Delta2.cuda(device)
        self.left_landmarks_indx = self.left_landmarks_indx.cuda(device)
        self.right_landmarks_indx = self.right_landmarks_indx.cuda(device)
        self.landmark_indx_speads = self.landmark_indx_speads.cuda(device)
        self.half_warping_window = self.half_warping_window.cuda(device)
        self.prec_ltri = self.prec_ltri.cuda(device)
        self.sigma_ltri = self.sigma_ltri.cuda(device)
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
        self.Delta2 = self.Delta2.cpu()
        self.left_landmarks_indx = self.left_landmarks_indx.cpu()
        self.right_landmarks_indx = self.right_landmarks_indx.cpu()
        self.landmark_indx_speads = self.landmark_indx_speads.cpu()
        self.half_warping_window = self.half_warping_window.cpu()
        self.prec_ltri = self.prec_ltri.cpu()
        self.sigma_ltri = self.sigma_ltri.cpu()
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
        ltri_matrix = torch.zeros(n_dims, n_dims, device=device)
        ltri_matrix[torch.arange(n_dims), torch.arange(n_dims)] = self.trial_peak_offset_covar_ltri_diag
        indices = torch.tril_indices(row=n_dims, col=n_dims, offset=-1)
        ltri_matrix[indices[0], indices[1]] = self.trial_peak_offset_covar_ltri_offdiag
        # We will call this Sigma
        self.sigma_ltri = ltri_matrix
        self.prec_ltri = torch.linalg.cholesky(torch.linalg.inv(ltri_matrix @ ltri_matrix.t()))


    def unnormalized_log_factors(self):
        # return self.beta - self.beta[:, 0].unsqueeze(1).expand_as(self.beta)
        return torch.cat([torch.zeros(self.n_factors, 1, device=self.device), self.beta], dim=1)


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
        W_AL = torch.sum(W_CKL, dim=(0, 1)).reshape(self.n_areas, -1)
        W_CKAL = W_CKL.reshape(W_CKL.shape[0], W_CKL.shape[1], self.n_areas, -1)
        for i in range(self.n_areas):
            if torch.all(W_AL[i] < tol):
                W_CKAL[:, :, i, :] = tol  # bandaid fix
                continue
            while torch.any(W_AL[i] < tol):
                min_idx = torch.argmin(W_AL[i])  # index of min population factor
                max_idx = torch.argmax(W_AL[i])  # index of max population factor
                highest_assignmet = torch.max(W_CKAL[:, :, i, max_idx])  # highest assignment to max factor
                max_members_idx = torch.where(W_CKAL[:, :, i, max_idx] == highest_assignmet)
                singele_max_member_idx = [max_members_idx[0][0], max_members_idx[1][0]]
                frac_to_move = tol if highest_assignmet/1000 > tol else highest_assignmet/1000
                W_CKAL[singele_max_member_idx[0], singele_max_member_idx[1], i, max_idx] -= frac_to_move
                W_CKAL[singele_max_member_idx[0], singele_max_member_idx[1], i, min_idx] += frac_to_move
                W_AL = torch.sum(W_CKAL, dim=(0, 1))
        self.W_CKL = W_CKAL.reshape(*W_CKL.shape[:2], -1)


    def generate_trial_peak_offset_samples(self):
        if self.is_eval:
            # trial_peak_offset_proposal_samples 1 x R x C x 2AL
            self.trial_peak_offset_proposal_samples = self.trial_peak_offset_proposal_means.unsqueeze(0)
        else:
            gaussian_sample = torch.concat([torch.randn(self.n_trial_samples, self.n_trials, self.n_configs, 2 * self.n_factors,
                                                        device=self.device),
                                            torch.zeros(1, self.n_trials, self.n_configs, 2 * self.n_factors,
                                                        device=self.device)], dim=0)
            # trial_peak_offset_proposal_samples 1+N x R x C x 2AL
            self.trial_peak_offset_proposal_samples = (self.trial_peak_offset_proposal_means.unsqueeze(0) +
                                                       gaussian_sample * self.trial_peak_offset_proposal_sds.unsqueeze(0))


    def warp_all_latent_factors_for_all_trials(self):
        return self.compute_warped_factors(self.compute_warped_times(*self.compute_offsets_and_landmarks()))


    def compute_offsets_and_landmarks(self):
        factors = torch.cat([torch.exp(self.unnormalized_log_factors())] * 2, dim=0)
        avg_peak_times = self.time[torch.tensor([self.left_landmarks_indx[i] + torch.argmax(factors[i, self.left_landmarks_indx[i]:self.right_landmarks_indx[i]])
                                                 for i in range(2*self.n_factors)])]
        left_landmarks = self.time[self.left_landmarks_indx]
        right_landmarks = self.time[self.right_landmarks_indx]
        avg_peak_times = torch.max(torch.stack([avg_peak_times, left_landmarks + self.dt], dim=0), dim=0).values
        avg_peak_times = torch.min(torch.stack([avg_peak_times, right_landmarks - self.dt], dim=0), dim=0).values
        # avg_peak_times  # 1 x 1 x 1 x 2AL
        avg_peak_times = avg_peak_times.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.generate_trial_peak_offset_samples()
        # trial_peak_offset_proposal_samples # N x R x C x 2AL
        # config_offsets  # 1 x 1 x C x 2AL
        s_new = avg_peak_times + self.trial_peak_offset_proposal_samples + self.config_peak_offsets.unsqueeze(0).unsqueeze(1)
        left_landmarks = left_landmarks.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        right_landmarks = right_landmarks.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        squash_mask = ((s_new <= left_landmarks) | (s_new >= right_landmarks)).to(device=self.device)
        s_new[squash_mask] = F.tanh(s_new[squash_mask]) * self.half_warping_window.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand_as(s_new)[squash_mask]
        return avg_peak_times, left_landmarks, right_landmarks, s_new


    def compute_warped_times(self, avg_peak_times, left_landmarks, right_landmarks, trial_peak_times):
        max_landmark_spread = self.landmark_indx_speads.max()
        # shifted_peak_times # 2AL x 1 x N x R x C
        left_shifted_peak_times = (trial_peak_times - left_landmarks).permute(3, 0, 1, 2).unsqueeze(1)
        right_shifted_peak_times = (trial_peak_times - right_landmarks).permute(3, 0, 1, 2).unsqueeze(1)
        # shifted_peak_times # 2AL x max_landmark_spread x N x R x C
        left_shifted_peak_times = left_shifted_peak_times.expand(-1, max_landmark_spread, -1, -1, -1)
        left_landmarks = left_landmarks.permute(3, 0, 1, 2).unsqueeze(1).expand_as(left_shifted_peak_times)
        right_landmarks = right_landmarks.permute(3, 0, 1, 2).unsqueeze(1).expand_as(left_shifted_peak_times)
        avg_peak_times = avg_peak_times.permute(3, 0, 1, 2).unsqueeze(1).expand_as(left_shifted_peak_times)
        left_slope = (avg_peak_times - left_landmarks) / left_shifted_peak_times
        right_slope = (avg_peak_times - right_landmarks) / right_shifted_peak_times
        # left_shifted_time # 2AL x max_landmark_spread x N x R x C
        left_shifted_time = torch.stack([torch.nn.functional.pad(self.time[:self.landmark_indx_speads[i]], (0, max_landmark_spread - self.landmark_indx_speads[i]),
                                                                 value=self.time[self.landmark_indx_speads[i]-1])
                                         for i in range(2*self.n_factors)]).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(left_shifted_peak_times)
        slope_mask = (left_shifted_time <= left_shifted_peak_times).int()
        # warped_times # 2AL x max_landmark_spread x N x R x C
        warped_times = slope_mask * (left_shifted_time * left_slope + left_landmarks) + (1 - slope_mask) * ((left_shifted_time - left_shifted_peak_times) * right_slope + avg_peak_times)
        return warped_times


    def compute_warped_factors(self, warped_times):
        factors = torch.cat([torch.exp(self.unnormalized_log_factors())] * 2, dim=0)
        # warped_time  # 2AL x len(landmark_spread) x N x R X C
        r0und = self.dt / (10 * int(str(self.dt.item())[-1]))
        warped_indices = (warped_times / self.dt) + r0und  # could be round but its not differentiable
        floor_warped_indices = warped_indices.int()  # could be torch.floor but for non-negative numbers it is the same
        ceil_warped_indices = (warped_indices+1).int()  # could be torch.ceil but for non-negative numbers it is the same
        ceil_weights = warped_indices - floor_warped_indices
        floor_weights = 1 - ceil_weights
        floor_warped_factor = torch.stack([factors[i, floor_warped_indices[i]] for i in range(2*self.n_factors)])
        weighted_floor_warped_factor = floor_warped_factor * floor_weights
        ceil_warped_factor = torch.stack([factors[i, ceil_warped_indices[i]] for i in range(2*self.n_factors)])
        weighted_ceil_warped_factor = ceil_warped_factor * ceil_weights
        peaks = weighted_floor_warped_factor + weighted_ceil_warped_factor
        factors = torch.exp(self.unnormalized_log_factors())
        early = [factors[l, :self.left_landmarks_indx[l]].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, *peaks.shape[2:]) for l in range(self.n_factors)]
        peak1 = [peaks[l, :self.landmark_indx_speads[l]] for l in range(self.n_factors)]
        mid = [factors[l, (self.right_landmarks_indx[l]+1):self.left_landmarks_indx[l+self.n_factors]].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, *peaks.shape[2:]) for l in range(self.n_factors)]
        peak2 = [peaks[l+self.n_factors, :self.landmark_indx_speads[l+self.n_factors]] for l in range(self.n_factors)]
        late = [factors[l, (self.right_landmarks_indx[l+self.n_factors]+1):].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, *peaks.shape[2:]) for l in range(self.n_factors)]
        # full_warped_factors # AL x T x N x R x C
        full_warped_factors = torch.stack([torch.cat([early[l], peak1[l], mid[l], peak2[l], late[l]], dim=0) for l in range(self.n_factors)])
        return full_warped_factors


    def prepare_inputs(self, processed_inputs):
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
        # trial_offsets # N x R x C x 2AL
        # neg_log_P # C x 1 x R x 1 x N
        neg_log_P = self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples, self.prec_ltri).unsqueeze(1).unsqueeze(3)
        # neg_log_Q # C x 1 x R x 1 x N
        neg_log_Q = self.sd_log_likelihood(self.trial_peak_offset_proposal_samples).unsqueeze(1).unsqueeze(3)
        elbo = (Y_times_warped_beta - Y_sum_t_times_logsumexp_warped_beta + (1 / R) * (alpha_times_log_theta_plus_b_CKL -
                    log_gamma_alpha) - (1 / K) * (neg_log_P - neg_log_Q))
        elbo = torch.sum(W_tensor * elbo)
        return elbo


    def Sigma_log_likelihood(self, trial_peak_offsets, ltri_matrix):
        n_dims = ltri_matrix.shape[0]  # 2AL
        inv_Sigma = ltri_matrix @ ltri_matrix.t() # Precision matrix
        prod_term = torch.einsum('nrcl,lj,nrcj->crn', trial_peak_offsets, inv_Sigma, trial_peak_offsets)  # sum over l
        entropy_term = 0.5 * (n_dims * torch.log(torch.tensor(2 * torch.pi)) - torch.logdet(inv_Sigma) + prod_term)
        return entropy_term # C x R x N


    def sd_log_likelihood(self, trial_peak_offsets):
        trial_peak_offsets = (trial_peak_offsets - self.trial_peak_offset_proposal_means.unsqueeze(0))**2
        n_dims = self.trial_peak_offset_proposal_sds.shape[-1]
        det_Sigma = torch.prod(self.trial_peak_offset_proposal_sds**2, dim=-1)
        inv_Sigma = self.trial_peak_offset_proposal_sds**(-2)
        prod_term = torch.sum(trial_peak_offsets * inv_Sigma.unsqueeze(0), dim=-1).permute(2, 1, 0) # sum over l
        entropy_term = 0.5 * (n_dims * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_Sigma.t().unsqueeze(-1)) + prod_term)
        return entropy_term  # C x R x N


    def compute_penalty_terms(self, tau_beta, tau_config, tau_sigma, tau_prec, tau_sd):
        # Penalty Terms
        self.ltri_matix()
        config_Penalty = - tau_config * (1/torch.prod(torch.tensor(self.config_peak_offsets.shape))) * self.config_peak_offsets.abs().sum()  # L1 penalty
        proposal_sd_penalty = - tau_sd * (1/torch.prod(torch.tensor(self.trial_peak_offset_proposal_sds.shape))) * (self.trial_peak_offset_proposal_sds**2).sum()  # L2 penalty
        sigma = self.sigma_ltri @ self.sigma_ltri.t()
        precision = self.prec_ltri @ self.prec_ltri.t()
        prec_Penalty = -tau_prec * (1/(torch.prod(torch.tensor(precision.shape))-precision.shape[0])) * precision.abs().sum() - precision.diag().abs().sum()  # L1 penalty
        sigma_Penalty = -tau_sigma * (1/sigma.shape[0]) * (sigma.diag()**2).sum()  # L2 penalty
        factors = torch.softmax(self.unnormalized_log_factors(), dim=-1)
        beta_penalty = -tau_beta * (1/torch.prod(torch.tensor(factors.shape))) * ((factors @ self.Delta2TDelta2) * factors).sum()
        penalty_term = config_Penalty + prec_Penalty + sigma_Penalty + beta_penalty + proposal_sd_penalty
        return penalty_term


    def forward(self, processed_inputs, update_membership=True, train=True):
        self.train(train)
        self.ltri_matix()
        processed_inputs = self.prepare_inputs(processed_inputs)
        if update_membership:
            self.E_step_posterior_updates(processed_inputs)
        return self.ELBO_term(processed_inputs)


    def log_likelihood(self, processed_inputs, E_step=False):
        self.train(False)
        self.ltri_matix()
        if E_step:
            processed_inputs = self.prepare_inputs(processed_inputs)
            self.E_step_posterior_updates(processed_inputs)
        # log_P  C x R
        log_P = -self.Sigma_log_likelihood(self.trial_peak_offset_proposal_samples, self.prec_ltri).squeeze()
        log_likelihood = torch.sum(self.log_sum_exp_U_tensor) + torch.sum(log_P)
        return log_likelihood


    def infer_latent_variables(self, processed_inputs):
        # likelihoods # C x K x L
        neuron_factor_assignment = torch.round(F.softmax(self.W_CKL.reshape(self.W_CKL.shape[0], self.W_CKL.shape[1], self.n_areas, -1)*1e5, dim=-1).reshape(self.W_CKL.shape), decimals=1)
        neuron_factor_assignment = neuron_factor_assignment * processed_inputs['neuron_factor_access']
        neuron_firing_rates = torch.sum(self.a_CKL * self.W_CKL, dim=-1)
        return neuron_factor_assignment, neuron_firing_rates