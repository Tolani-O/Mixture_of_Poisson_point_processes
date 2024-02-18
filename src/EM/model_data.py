import numpy as np
from scipy.interpolate import BSpline


class ModelData:

    def __init__(self):
        self.time = None
        self.joint_factors_indices = None
        self.degree = None
        self.dt = None
        self.knots = None
        self.B = None
        self.left_landmark1 = 20
        self.mid_landmark1 = 45
        self.right_landmark1 = 70
        self.left_landmark2 = 120
        self.mid_landmark2 = 145
        self.right_landmark2 = 170
        self.Y = None
        self.trial_warped_factors = None
        self.trial_warped_splines = None
        self.trial_peak_offsets = None

        #parameters
        self.beta = None  # AL x P
        self.alpha = None  # 1 x AL
        self.theta = None  # 1 x AL

    def initialize(self, time, joint_factor_indices, degree=3):
        self.time = time
        self.joint_factors_indices = joint_factor_indices
        self.degree = degree
        self.dt = round(self.time[1] - self.time[0], 3)
        self.knots = np.concatenate([np.repeat(self.time[0], degree), self.time, np.repeat(self.time[-1], degree)])
        self.knots[-1] = self.knots[-1] + self.dt
        self.B = BSpline.design_matrix(self.time, self.knots, degree).transpose()

    def randomly_initialize_parameters(self):
        self.alpha = np.random.uniform(0, 1, self.beta.shape[0])
        self.theta = np.random.uniform(0, 1, self.beta.shape[0])

    def warp_all_latent_factors_for_all_trials(self):
        factors = np.exp(self.beta) @ self.B
        trial_warped_factors = []
        trial_warped_splines = []
        L = factors.shape[0]
        R = self.trial_peak_offsets.shape[0]
        for l in range(L):
            if l in self.joint_factors_indices:
                # cross trial average peak times
                avg_peak_time1 = self.time[
                    self.left_landmark1 + np.argmax(factors[l, self.left_landmark1:self.right_landmark1])]
                avg_peak_time2 = self.time[
                    self.left_landmark2 + np.argmax(factors[l, self.left_landmark2:self.right_landmark2])]
            else:
                avg_peak_time1 = self.time[self.mid_landmark1]
                avg_peak_time2 = self.time[self.mid_landmark2]
            trial_factors = []
            trial_splines = []
            for r in range(R):
                warped_splines, warped_factor = (
                    self.warp_latent_factor_for_trial(np.exp(self.beta[l, :]), [avg_peak_time1, avg_peak_time2],
                                                      self.trial_peak_offsets[r, 2 * l:2 * (l + 1)]))
                trial_factors.append(warped_factor)
                trial_splines.append(warped_splines)
            trial_warped_factors.append(np.vstack(trial_factors))
            trial_warped_splines.append(trial_splines)
        self.trial_warped_factors = np.stack(trial_warped_factors)
        self.trial_warped_splines = trial_warped_splines

    def warp_latent_factor_for_trial(self, gamma, avg_peak_times, trial_peak_times):  # warped factors for trial r
        warped_time = self.warp_time_for_trial(avg_peak_times, trial_peak_times)
        warped_splines = BSpline.design_matrix(warped_time, self.knots, self.degree).transpose()
        warped_factor = gamma @ warped_splines
        return warped_splines, warped_factor

    def warp_time_for_trial(self, avg_peak_times, trial_peak_times):
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
            if t < l1:
                warped_time[i] = t
            elif t < s1_new:
                warped_time[i] = (t - l1) * (p1 - l1) / (s1_new - l1) + l1
            elif t < r1:
                warped_time[i] = (t - s1_new) * (r1 - p1) / (r1 - s1_new) + p1
            elif t < l2:
                warped_time[i] = t
            elif t < s2_new:
                warped_time[i] = (t - l2) * (p2 - l2) / (s2_new - l2) + l2
            elif t < r2:
                warped_time[i] = (t - s2_new) * (r2 - p2) / (r2 - s2_new) + p2
            else:
                warped_time[i] = t
        return warped_time

    def compute_log_likelihood(self):
        pass
