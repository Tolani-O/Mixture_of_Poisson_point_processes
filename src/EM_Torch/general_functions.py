import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.figure import figaspect
import json
import argparse
from torch.utils.data import Dataset
from tslearn.clustering import TimeSeriesKMeans
from scipy.ndimage import gaussian_filter1d
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
import pickle
sns.set()
plt.rcParams.update({'figure.max_open_warning': 0})

def get_parser():
    parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
    parser.add_argument('--folder_name', type=str, default='', help='folder name')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA (default: True)')
    parser.add_argument('--time_warp', type=bool, default=True, help='whether or not to use time warping')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials per stimulus condition')
    parser.add_argument('--n_configs', type=int, default=2, help='Number of stimulus conditions')
    parser.add_argument('--A', type=int, default=2, help='Number of areas')
    parser.add_argument('--n_trial_samples', type=int, default=10, help='Number of trial samples for monte carlo integration')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
    parser.add_argument('--load', type=int, default=0, help='')
    parser.add_argument('--load_epoch', type=int, default=-1, help='Which epoch to load model and optimizer from. '
                                                                   'Default is -1 meaning it will load the latest model')
    parser.add_argument('--load_run', type=int, default=0, help='Which run to load model and optimizer from')
    parser.add_argument('--temperature', type=float, default=1, help='Softmax temperature')
    parser.add_argument('--weights', type=float, default=1, help='temperature weights (for multiple temperatures)')
    parser.add_argument('--mask_neuron_threshold', type=int, default=0, help='If neuron spike count is below this, mask neuron')
    parser.add_argument('--tau_config', type=float, default=0.5, help='Value for tau_config')
    parser.add_argument('--tau_sigma', type=float, default=0.5, help='Value for tau_sigma')
    parser.add_argument('--tau_sd', type=float, default=0.5, help='Value for tau_sd')
    parser.add_argument('--tau_beta', type=float, default=0.5, help='Value for tau_beta')
    parser.add_argument('--num_epochs', type=int, default=-1, help='Number of training epochs. '
                                                                   'Default is -1 meaning it will run the hessian computation')
    parser.add_argument('--scheduler_patience', type=int, default=1e5, help='Number of epochs before scheduler step')
    parser.add_argument('--scheduler_factor', type=int, default=0.9, help='Scheduler reduction factor')
    parser.add_argument('--scheduler_threshold', type=int, default=1e-10, help='Threshold to accept step improvement')
    parser.add_argument('--notes', type=str, default='', help='Run notes')
    parser.add_argument('--K', type=int, default=30, help='Number of neurons')
    parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
    parser.add_argument('--landmark_step', type=float, default=1, help='step size for time warping landmarks')
    parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
    parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
    parser.add_argument('--param_seed', type=int_or_str, default='', help='options are: seed (int), Truth (str)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval (default: 100')
    parser.add_argument('--eval_interval', type=int, default=100, metavar='N', help='report interval (default: 10')
    parser.add_argument('--init', type=str, default='dtw', help='initialization for the algorithm. options are: '
                                                                'dtw, mom, rand, zero, true')
    return parser


class CustomDataset(Dataset):
    def __init__(self, Y, neuron_factor_access):
        # Y # K x T x R x C
        # neuron_factor_access  #  C x K x L
        self.Y = Y
        self.neuron_factor_access = neuron_factor_access

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.Y[idx], self.neuron_factor_access[:, idx, :] # K x C x L


def preprocess_input_data(Y, neuron_factor_access, dt, mask_threshold=0):
    processed_inputs = {}
    # Y # K x T x R x C
    # neuron_factor_access # C x K x L
    Y_sum_t = torch.sum(Y, dim=1).permute(2, 0, 1)  # C x K x R
    Y_sum_rt = torch.sum(Y_sum_t, dim=-1)  # C x K
    empty_trials = torch.where(Y_sum_rt <= mask_threshold)
    neuron_factor_access[empty_trials[0], empty_trials[1]] = 0
    time = torch.arange(Y.shape[1], device=Y.device) * dt
    processed_inputs['time'] = time
    processed_inputs['Y'] = Y
    processed_inputs['Y_sum_t'] = Y_sum_t
    processed_inputs['Y_sum_rt'] = Y_sum_rt
    processed_inputs['neuron_factor_access'] = neuron_factor_access
    return processed_inputs


def plot_spikes(binned, output_dir, dt, filename, x_offset=0):
    # Group entries by unique values of s[0]
    Y = np.vstack(np.transpose(binned, (2,1,0,3)))
    Y = np.vstack(np.transpose(Y, (2,1,0)))
    spikes = np.where(Y >= 1)
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset) * dt
        grouped_s.append((i, values))
    aspect_ratio = Y.shape[0] / Y.shape[1]
    w, h = figaspect(aspect_ratio)
    plt.figure(figsize=(w, h))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.savefig(os.path.join(output_dir, f'groundTruth_spikes_{filename}.png'))


def plot_intensity_and_latents(time, latent_factors, intensity, output_dir):
    # plot latent factors
    plt.figure()
    for i in range(latent_factors.shape[0]):
        plt.plot(time, latent_factors[i, :] + i)
    plt.savefig(os.path.join(output_dir, 'groundTruth_latent_factors.png'))

    # plot neuron intensities
    Y = np.vstack(np.transpose(intensity, (2, 1, 0, 3)))
    Y = np.vstack(np.transpose(Y, (2, 1, 0)))
    dt = time[1] - time[0]
    K, T = Y.shape
    plt.figure()
    for i in range(K):
        plt.plot(np.arange(T) * dt, Y[i] + i*1)
    plt.savefig(os.path.join(output_dir, 'groundTruth_intensities.png'))


def plot_latent_coupling(latent_coupling, output_dir):
    # plot latent couplings
    plt.figure(figsize=(10, 10))
    sns.heatmap(latent_coupling, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of clusters')
    plt.savefig(os.path.join(output_dir, f'groundTruth_neuron_clusters.png'))


def load_model_checkpoint(output_dir, load_epoch):
    if load_epoch < 0:
        with open(os.path.join(output_dir, 'json', 'epoch_train.json'), 'r') as f:
            load_epoch = json.load(f)[-1]
    print(f'Loading model from epoch {load_epoch}')
    load_model_dir = os.path.join(output_dir, 'models', f'model_{load_epoch}.pth')
    intermediate_vars_dir = os.path.join(output_dir, 'models', f'intermediate_vars_{load_epoch}.pkl')
    load_optimizer_dir = os.path.join(output_dir, 'models', f'optimizer_{load_epoch}.pth')
    scheduler_dir = os.path.join(output_dir, 'models', f'scheduler_{load_epoch}.pth')
    if os.path.isfile(load_model_dir):
        model = torch.load(load_model_dir, map_location=torch.device('cpu'))
    else:
        raise Exception(f'No model_{load_epoch}.pth file found at {load_model_dir}')
    if os.path.isfile(intermediate_vars_dir):
        with open(intermediate_vars_dir, 'rb') as f:
            intermediate_vars = pickle.load(f)
            W_CKL, a_CKL, theta, pi = intermediate_vars['W_CKL'], intermediate_vars['a_CKL'], intermediate_vars['theta'], intermediate_vars['pi']
    else:
        raise Exception(f'No intermediate_vars_{load_epoch}.pkl file found at {intermediate_vars_dir}')
    if os.path.isfile(load_optimizer_dir):
        optimizer = torch.load(load_optimizer_dir, map_location=torch.device('cpu'))
    else:
        raise Exception(f'No optimizer_{load_epoch}.pth file found at {load_optimizer_dir}')
    if os.path.isfile(scheduler_dir):
        scheduler = torch.load(scheduler_dir, map_location=torch.device('cpu'))
    else:
        raise Exception(f'No scheduler_{load_epoch}.pth file found at {scheduler_dir}')
    return model, optimizer, scheduler, W_CKL, a_CKL, theta, pi, load_epoch


def reset_metric_checkpoint(output_dir, folder_name, sub_folder_name, metric_files, start_epoch):
    metrics_dir = os.path.join(output_dir, folder_name, sub_folder_name)
    for metric_file in metric_files:
        path = os.path.join(metrics_dir, f'{metric_file}.json')
        with open(path, 'rb') as file:
            file_contents = json.load(file)
        if len(file_contents) > start_epoch:
            # Keep only the first num_keep_entries
            file_contents = file_contents[:start_epoch]
        # Write the modified data back to the file
        with open(path, 'w') as file:
            json.dump(file_contents, file, indent=4)


def create_relevant_files(output_dir, output_str, params=None, ground_truth=False):
    with open(os.path.join(output_dir, 'log.txt'), 'w') as file:
        file.write(output_str)

    output_dir = os.path.join(output_dir, 'json')
    os.makedirs(output_dir, exist_ok=True)
    if params is not None:
        with open(os.path.join(output_dir, 'params.json'), 'w') as file:
            json.dump(params, file, indent=4)

    # List of JSON files to be created
    file_names = [
        'log_likelihoods_batch',
        'losses_batch',
        'epoch_batch',
        'log_likelihoods_train',
        'true_log_likelihoods_train',
        'losses_train',
        'epoch_train',
        'alpha_batch_grad_norms',
        'beta_batch_grad_norms',
        'config_peak_offsets_batch_grad_norms',
        'trial_peak_offset_covar_ltri_diag_batch_grad_norms',
        'trial_peak_offset_covar_ltri_offdiag_batch_grad_norms',
        'trial_peak_offset_proposal_means_batch_grad_norms',
        'trial_peak_offset_proposal_sds_batch_grad_norms',
        'alpha_train_grad_norms',
        'beta_train_grad_norms',
        'config_peak_offsets_train_grad_norms',
        'trial_peak_offset_covar_ltri_diag_train_grad_norms',
        'trial_peak_offset_covar_ltri_offdiag_train_grad_norms',
        'trial_peak_offset_proposal_means_train_grad_norms',
        'trial_peak_offset_proposal_sds_train_grad_norms'
    ]
    if ground_truth:
        test_file_names = [
            'log_likelihoods_test',
            'losses_test',
            'beta_MSE_test',
            'alpha_MSE_test',
            'theta_MSE_test',
            'pi_MSE_test',
            'proposal_means_MSE_test',
            'configoffset_MSE_test',
            'ltri_MSE_test',
            'Sigma_MSE_test',
            'ltriLkhd_train',
            'ltriLkhd_test',
            'gains_MSE_train',
            'gains_MSE_test',
        ]
        file_names.extend(test_file_names)

    # Iterate over the filenames and create each file
    for file_name in file_names:
        with open(os.path.join(output_dir, '.'.join([file_name, 'json'])), 'w+b') as file:
            file.write(b'[]')


def write_log_and_model(output_str, output_dir, epoch, model, optimizer, scheduler):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
        file.write(output_str)
    models_path = os.path.join(output_dir, 'models')
    os.makedirs(models_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_path, f'model_{epoch}.pth'))
    with open(os.path.join(models_path, f'intermediate_vars_{epoch}.pkl'), 'wb') as f:
        pickle.dump({'W_CKL': model.W_CKL, 'a_CKL': model.a_CKL, 'theta': model.theta, 'pi': model.pi}, f)
    torch.save(optimizer.state_dict(), os.path.join(models_path, f'optimizer_{epoch}.pth'))
    torch.save(scheduler.state_dict(), os.path.join(models_path, f'scheduler_{epoch}.pth'))


def parse_folder_name(folder_name, parser_key, outputs_folder, load_run):
    parsed_values = {}
    for key in parser_key:
        start_idx = folder_name.find(key)
        if start_idx != -1:
            end_idx = folder_name.find('_', start_idx)
            if end_idx == -1:
                end_idx = len(folder_name)
            value = folder_name[(start_idx + len(key)):end_idx]
            parsed_values[key] = value
    load_dir = os.path.join(os.getcwd(), outputs_folder, folder_name, f'Run_{load_run}', 'json', 'params.json')
    if os.path.exists(load_dir):
        with open(load_dir, 'r') as f:
            loaded_params = json.load(f)
        for key in loaded_params:
            if key not in parsed_values:
                parsed_values[key] = loaded_params[key]
    return parsed_values


def plot_outputs(model, unique_regions, output_dir, folder, epoch, se_dict=None, Y=None, factor_access=None, warp_data=True, reorder_factors=True):
    model.train(False)
    stderr = se_dict is not None
    plot_data = (Y is not None) and (model.W_CKL is not None)
    output_dir = os.path.join(output_dir, folder)
    os.makedirs(output_dir, exist_ok=True)
    beta_dir = os.path.join(output_dir, 'beta')
    os.makedirs(beta_dir, exist_ok=True)
    warp_time_dir = os.path.join(output_dir, 'warped_times')
    os.makedirs(warp_time_dir, exist_ok=True)
    log_beta_dir = os.path.join(output_dir, 'log_beta')
    os.makedirs(log_beta_dir, exist_ok=True)
    alpha_dir = os.path.join(output_dir, 'alpha')
    os.makedirs(alpha_dir, exist_ok=True)
    theta_dir = os.path.join(output_dir, 'theta')
    os.makedirs(theta_dir, exist_ok=True)
    pi_dir = os.path.join(output_dir, 'pi')
    os.makedirs(pi_dir, exist_ok=True)
    configoffset_dir = os.path.join(output_dir, 'configoffset')
    os.makedirs(configoffset_dir, exist_ok=True)
    ltri_dir = os.path.join(output_dir, 'ltri')
    os.makedirs(ltri_dir, exist_ok=True)
    pcorr_dir = os.path.join(output_dir, 'pcorr')
    os.makedirs(pcorr_dir, exist_ok=True)
    corr_dir = os.path.join(output_dir, 'corr')
    os.makedirs(corr_dir, exist_ok=True)
    proposal_means_dir = os.path.join(output_dir, 'proposal_means')
    os.makedirs(proposal_means_dir, exist_ok=True)
    trial_sd_dir = os.path.join(output_dir, 'trial_SDs')
    os.makedirs(trial_sd_dir, exist_ok=True)
    with torch.no_grad():
        beta = model.unnormalized_log_factors().numpy()
        L = beta.shape[0]
        if stderr:
            beta_se = torch.cat([torch.zeros(L, 1), se_dict['beta']], dim=1).numpy() * 1.96
            for landmarks in [model.left_landmarks_indx, model.right_landmarks_indx]:
                beta_se[np.array([range(L)]*2).flatten(), landmarks] = (beta_se[range(L), landmarks - 1] +
                                                 beta_se[range(L), landmarks + 1]) / 2
            beta_se = np.where(beta_se == 0, (np.roll(beta_se, 1, axis=1) + np.roll(beta_se, -1, axis=1)) / 2, beta_se)
            beta_ucl = beta + beta_se
            beta_lcl = beta - beta_se
        upper_limit = np.max(beta) + 0.1
        lower_limit = np.min(beta) - 0.01
        plt.figure(figsize=(10, L*5))
        for l in range(L):
            plt.subplot(L, 1, l + 1)
            plt.plot(model.time, beta[l, :], label=f'Log Factor {l}')
            if stderr:
                plt.fill_between(model.time, beta_ucl[l, :], beta_lcl[l, :], color='grey', alpha=0.3,  label='Standard Error')
                plt.plot(model.time, beta_ucl[l, :], linestyle='--', color='black', alpha=0.07)
                plt.plot(model.time, beta_lcl[l, :], linestyle='--', color='black', alpha=0.07)
                plt.legend()
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Log Factor [{l}, :]')
            plt.ylim(bottom=lower_limit, top=upper_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(log_beta_dir, f'beta_{epoch}.png'))
        plt.close()

        pi = model.pi.numpy()
        if model.W_CKL is None:
            W_L = np.zeros(model.n_factors)
        else:
            plot_factor_assignments(model.W_CKL.numpy(), output_dir, 'cluster', epoch, False)
            W_L = np.round((torch.sum(model.W_CKL, dim=(0, 1))/model.n_configs).numpy(), 1)
        beta = model.unnormalized_log_factors()
        latent_factors = F.softmax(beta, dim=-1).numpy()
        upper_limit = np.max(latent_factors) + 0.005
        lower_limit = -5e-4
        if plot_data:
            neuron_factor_assignment, neuron_firing_rates = model.infer_latent_variables({'neuron_factor_access': factor_access})
            K, T, R, C = Y.shape
            data = Y / (1e-10 + neuron_firing_rates.t().unsqueeze(1).unsqueeze(2))
            # scaled_data L x C x R x T
            if warp_data:
                data = reverse_warp_data(model, data)
            else:
                data = torch.einsum('ktrc,ckl->lcrt', data, model.W_CKL)
            scaled_data = data.sum(dim=(1, 2))/(R * model.W_CKL.sum(dim=(0, 1)).unsqueeze(-1)).numpy()
        if stderr:
            softmax_grad = np.abs(latent_factors * (1 - latent_factors))
            latent_factors_se = softmax_grad * beta_se
            factor_ucl = latent_factors + latent_factors_se
            factor_lcl = latent_factors - latent_factors_se
        factors_per_area = int(model.n_factors / model.n_areas)
        L = np.arange(model.n_factors).reshape(model.n_areas, -1)
        ordr = np.concatenate([np.arange(factors_per_area)] * model.n_areas).reshape(model.n_areas, -1)
        if reorder_factors:
            ordr = W_L.reshape(model.n_areas, -1).argsort()
        indcs = np.concatenate([L[i, ordr[i][::-1]] for i in range(model.n_areas)]).reshape(model.n_areas, -1)
        L = indcs.T.flatten()
        plt.figure(figsize=(model.n_areas*10, int(model.n_factors/model.n_areas)*5))
        c = 0
        for l in L:
            plt.subplot(factors_per_area, model.n_areas, c + 1)
            plt.vlines(x=model.time[torch.tensor([
                model.left_landmarks_indx[l], model.right_landmarks_indx[l],
                model.left_landmarks_indx[l+model.n_factors], model.right_landmarks_indx[l+model.n_factors]])],
                       ymin=0, ymax=upper_limit,
                       color='grey', linestyle='--', alpha=0.5)
            plt.hlines(y=0, xmin=0, xmax=model.time[-1], color='white', linestyle='-', linewidth=2, alpha=0.5)
            plt.plot(model.time, latent_factors[l, :], alpha=np.max([pi[l], 0.5]), linewidth=np.exp(1.5*pi[l]))
            if plot_data:
                plt.plot(model.time, scaled_data[l, :], alpha=np.max([pi[l], 0.5]), linewidth=1, linestyle='--')
            if stderr:
                plt.fill_between(model.time, factor_ucl[l, :], factor_lcl[l, :], color='grey', alpha=0.3, label='Standard Error')
                plt.plot(model.time, factor_ucl[l, :], linestyle='--', color='black', alpha=0.07)
                plt.plot(model.time, factor_lcl[l, :], linestyle='--', color='black', alpha=0.07)
                plt.legend()
            plt.xlabel('Intensity')
            plt.ylabel('Trial time course (ms)')
            plt.title(f'Factor {(l%factors_per_area)+1}, '
                      f'Area {unique_regions[l//factors_per_area]}, '
                      f'Membership: {pi[l]:.2f}, '
                      f'Count: {W_L[l]:.1f}', fontsize=20)
            plt.ylim(bottom=lower_limit, top=upper_limit)
            c += 1
        plt.tight_layout()
        plt.savefig(os.path.join(beta_dir, f'LatentFactors_{epoch}.png'))
        plt.close()

        avg_peak_times, left_landmarks, right_landmarks, s_new = model.compute_offsets_and_landmarks()
        warped_times = model.compute_warped_times(avg_peak_times, left_landmarks, right_landmarks, s_new)
        warped_times = warped_times.squeeze().reshape(*warped_times.shape[:2], -1)
        avg_peak_times = avg_peak_times.squeeze()
        expended_time = model.time.unsqueeze(-1).expand(-1, warped_times.shape[-1])
        plt.figure(figsize=(model.n_areas * 10, int(model.n_factors / model.n_areas) * 5))
        c = 0
        for l in L:
            plt.subplot(factors_per_area, model.n_areas, c + 1)
            warped_times_l = torch.concat([expended_time[:model.left_landmarks_indx[l]],
                                           warped_times[l, :model.landmark_indx_speads[l]],
                                           expended_time[(model.right_landmarks_indx[l]+1):model.left_landmarks_indx[l+model.n_factors]],
                                           warped_times[l+model.n_factors, :model.landmark_indx_speads[l+model.n_factors]],
                                           expended_time[(model.right_landmarks_indx[l+model.n_factors]+1):]],
                                          dim=0).t().numpy()
            latent_factors_l = latent_factors[l]
            lf_max, lf_min = latent_factors_l.max(), latent_factors_l.min()
            latent_factors_l = (latent_factors_l - lf_min) * (model.time[-1].item() * (4/5)) / (lf_max - lf_min)
            for xx in warped_times_l:
                plt.plot(model.time, xx, color='grey', alpha=0.2)
            plt.plot(model.time, model.time, color='black')
            plt.plot(model.time, latent_factors_l, color='blue', linestyle='--', alpha=0.3)
            plt.vlines(x=[avg_peak_times[l], avg_peak_times[l+model.n_factors]], ymin=0, ymax=model.time[-1], color='grey', linestyle='--', alpha=0.5)
            plt.xlabel('Warped time')
            plt.ylabel('Time (ms)')
            plt.title(f'Factor {(l % factors_per_area) + 1}, '
                      f'Area {unique_regions[l // factors_per_area]}, '
                      f'warp', fontsize=20)
            c += 1
        plt.tight_layout()
        plt.savefig(os.path.join(warp_time_dir, f'warped_times_{epoch}.png'))
        plt.close()

        proposal_means = model.transform_peak_offsets().squeeze().permute(2, 0, 1).reshape(warped_times.shape[0], -1)
        plt.figure(figsize=(model.n_areas * 15, int(model.n_factors / model.n_areas) * 5))
        c = 0
        xlimit = proposal_means.abs().max().item()
        for l in L:
            for p in range(2):
                i = l + p * model.n_factors
                plt.subplot(factors_per_area, 2 * model.n_areas, c + 1)
                counts, bin_edges = np.histogram(proposal_means[i], bins=30)
                plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7)
                plt.xlabel('Trial peak times')
                plt.ylabel('Frequency')
                plt.xlim(left=-xlimit, right=xlimit)
                # plt.ylim(bottom=-1e-5, top=100)
                plt.title(f'Factor {(l % factors_per_area) + 1}, '
                          f'Area {unique_regions[l // factors_per_area]}, '
                          f'Peak {p + 1} offsets', fontsize=20)
                c += 1
        plt.tight_layout()
        plt.savefig(os.path.join(proposal_means_dir, f'proposal_means_{epoch}.png'))
        plt.close()

        ltri_matrix = model.ltri_matix('cpu').numpy()
        precision = ltri_matrix @ ltri_matrix.T
        srt = np.concatenate([indcs.flatten(), indcs.flatten() + model.n_factors])
        precision = precision[srt].T[srt]
        partial_correlation = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
        diag_indcs = np.arange(partial_correlation.shape[0])
        partial_correlation[diag_indcs, diag_indcs] = np.abs(np.diag(partial_correlation))
        plt.figure(figsize=(10, 10))
        factors_per_area = model.n_factors // model.n_areas
        ax = sns.heatmap(partial_correlation, annot=False, cmap="seismic", center=0, vmin=-1, vmax=1,
                         xticklabels=factors_per_area, yticklabels=factors_per_area)
        for i in range(factors_per_area, partial_correlation.shape[0], factors_per_area):
            ax.axvline(i, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(i, color='black', linestyle='-', linewidth=0.5)
        plt.title('Peak time partial correlation matrix')
        plt.savefig(os.path.join(pcorr_dir, f'pcorr_{epoch}.png'))
        plt.close()

        covariance = np.linalg.inv(precision)
        correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        plt.figure(figsize=(10, 10))
        factors_per_area = model.n_factors // model.n_areas
        ax = sns.heatmap(correlation, annot=False, cmap="seismic", center=0, vmin=-1, vmax=1,
                         xticklabels=factors_per_area, yticklabels=factors_per_area)
        for i in range(factors_per_area, correlation.shape[0], factors_per_area):
            ax.axvline(i, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(i, color='black', linestyle='-', linewidth=0.5)
        plt.title('Peak time correlation matrix')
        plt.savefig(os.path.join(corr_dir, f'corr_{epoch}.png'))
        plt.close()

        alpha = F.softplus(model.alpha).numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(alpha, label='Alpha')
        plt.title('Alpha')
        plt.savefig(os.path.join(alpha_dir, f'alpha_{epoch}.png'))
        plt.close()

        theta = model.theta.numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(theta, label='Theta')
        plt.title('Theta')
        plt.savefig(os.path.join(theta_dir, f'theta_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(pi, label='Pi')
        plt.title('Pi')
        plt.savefig(os.path.join(pi_dir, f'pi_{epoch}.png'))
        plt.close()

        if plot_data:
            # scaled_data L x C x T
            scaled_data = data.sum(dim=2) / (R * model.W_CKL.sum(dim=1).t().unsqueeze(-1).numpy())
        # config_times L x C x 2
        config_times = avg_peak_times.unsqueeze(0) + model.transform_peak_offsets(config_offsets=True).numpy()
        config_times = config_times.reshape(model.n_configs, 2, -1).permute(2, 0, 1).numpy()
        plt.figure(figsize=(model.n_areas * 10, int(model.n_factors / model.n_areas) * 5))
        c = 0
        for l in L:
            plt.subplot(factors_per_area, model.n_areas, c + 1)
            plt.vlines(x=model.time[torch.tensor([
                model.left_landmarks_indx[l], model.right_landmarks_indx[l],
                model.left_landmarks_indx[l+model.n_factors], model.right_landmarks_indx[l+model.n_factors]])],
                       ymin=0, ymax=upper_limit,
                       color='grey', linestyle='--', alpha=0.5)
            plt.hlines(y=0, xmin=0, xmax=model.time[-1], color='white', linestyle='-', linewidth=2, alpha=0.5)
            plt.plot(model.time, latent_factors[l, :], alpha=np.max([pi[l], 0.5]),
                     linewidth=np.exp(1.5 * pi[l]))
            for cnf in range(model.n_configs):
                if plot_data:
                    plt.plot(model.time, scaled_data[l, cnf], color='grey', alpha=0.4, linewidth=1, linestyle='--')
                plt.vlines(x=config_times[l, cnf, 0], ymin=0, ymax=upper_limit,
                           color='red', linestyle='--', alpha=0.5)
                plt.vlines(x=config_times[l, cnf, 1], ymin=0, ymax=upper_limit,
                           color='green', linestyle='--', alpha=0.5)
            plt.xlabel('Intensity')
            plt.ylabel('Trial time course (ms)')
            plt.title(f'Factor {(l % factors_per_area) + 1}, '
                      f'Area {unique_regions[l // factors_per_area]}, '
                      f'Membership: {pi[l]:.2f}, '
                      f'Count: {W_L[l]:.1f}', fontsize=20)
            c += 1
        plt.tight_layout()
        plt.savefig(os.path.join(configoffset_dir, f'configoffset_{epoch}.png'))
        plt.close()

        ltri = model.ltri_matix('cpu').flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(ltri, label='Ltri')
        plt.title('Ltri')
        plt.savefig(os.path.join(ltri_dir, f'ltri_{epoch}.png'))
        plt.close()

        trial_SDs = model.trial_peak_offset_proposal_sds.flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(trial_SDs, label='Trial Standard Deviations')
        plt.title('Trial Standard Deviations')
        plt.savefig(os.path.join(trial_sd_dir, f'trial_variances_{epoch}.png'))
        plt.close()
        plt.close('all')


def initialize_clusters(Y, factor_access, n_clusters, n_areas, output_dir, n_jobs=15, bandwidth=4):
    # Y # K x T x R x C
    # factor_access  # C x K x L
    K, T, R, C = Y.shape
    Y_train = torch.tensor(gaussian_filter1d(Y.sum(axis=2), sigma=bandwidth, axis=1)).permute(2, 0, 1)  # C x K x T
    Y_train = 100 * Y_train / Y_train.sum(dim=-1).unsqueeze(-1)
    # data # C x K x T x A x L
    data = torch.einsum('ckt,ckl->cktl', Y_train, factor_access).reshape(C, K, T, n_areas, n_clusters)
    regions = [torch.cat(list(data[:, :, :, i, 0])) for i in range(n_areas)]
    indices = [torch.where(reg.sum(dim=-1) > 0)[0] for reg in regions]
    regions_active = [regn[indc] for regn, indc in zip(regions, indices)]
    dba_km = TimeSeriesKMeans(n_clusters=n_clusters, n_init=10, metric='dtw', max_iter_barycenter=20, n_jobs=n_jobs)
    neuron_factor_assignment = torch.zeros((K*C, n_clusters*n_areas), dtype=torch.float64)
    predicted_y = torch.zeros((C*K), dtype=torch.int64)
    region_log_factors = []
    print('Fitting DBA-KMeans')
    for i, regn_indc in enumerate(zip(regions_active, indices)):
        print(f'Fitting Area {i + 1}')
        regn, indc = regn_indc
        y_pred = dba_km.fit_predict(regn)
        neuron_factor_assignment[indc, (i*n_clusters)+y_pred] = 1
        beta = torch.log(torch.tensor(dba_km.cluster_centers_).squeeze())
        region_log_factors.append(beta)
        predicted_y[indc] = torch.tensor((i*n_clusters)+y_pred)
    neuron_factor_assignment = neuron_factor_assignment.reshape(C, K, -1)
    predicted_y = predicted_y.reshape(C, K)
    latent_factors = torch.cat(region_log_factors, dim=0)
    # save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, 'cluster_initialization.pkl')
    print('Saving clusters to: ', save_dir)
    with open(save_dir, 'wb') as f:
        pickle.dump({'y_pred': predicted_y, 'neuron_factor_assignment': neuron_factor_assignment, 'beta': latent_factors}, f)


def plot_initial_clusters(output_dir, data_folder, n_clusters, data=None):
    # Y # K x T x R x C
    if data is None:
        data_dir = os.path.join(output_dir, data_folder, f'{data_folder}.pkl')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        with open(data_dir, 'rb') as f:
            data = pickle.load(f)
    Y, time = torch.tensor(data['Y']), torch.tensor(data['time'])
    cluster_dir = os.path.join(output_dir, data_folder, 'cluster_initialization.pkl')
    if not os.path.exists(cluster_dir):
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
    with open(cluster_dir, 'rb') as f:
        cluster = pickle.load(f)
    y_pred, neuron_factor_assignment, beta = cluster['y_pred'], cluster['neuron_factor_assignment'], cluster['beta']
    y_pred = ((y_pred + 1) * neuron_factor_assignment.sum(dim=-1)).int()
    factors = F.softmax(beta, dim=-1)
    Y_train = Y.sum(axis=2).permute(2, 0, 1).float()  # C x K x T
    Y_train = F.softmax(Y_train, dim=-1)
    # Y_train = (Y_train / Y_train.sum(dim=-1).unsqueeze(-1)).nan_to_num(0) * neuron_factor_assignment.sum(dim=-1).unsqueeze(-1)
    n_factors = beta.shape[0]
    n_areas = n_factors // n_clusters
    plt.figure(figsize=(n_areas * 10, n_clusters * 5))
    L = np.arange(n_factors).reshape(n_areas, -1).T.flatten()
    c = 0
    factors_per_area = n_clusters
    upper_limit = torch.max(factors).item() + 0.005
    for l in L:
        plt.subplot(factors_per_area, n_areas, c + 1)
        units_indcs = torch.where(y_pred == (l+1))
        units = Y_train[units_indcs]
        for xx in units:
            plt.plot(time, xx.ravel(), "k-", alpha=0.2)
        plt.plot(time, factors[l, :], "r-")
        plt.xlabel('Intensity')
        plt.ylabel('Trial time course (ms)')
        plt.ylim(bottom=-5e-4, top=upper_limit)
        plt.title(f'Factor {(l % factors_per_area) + 1}, '
                  f'Area {(l // factors_per_area) + 1}, '
                  f'Count: {units.shape[0]:.1f}', fontsize=20)
        c += 1
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, data_folder, 'cluster_initialization.png'))
    plt.close()


def plot_factor_assignments(factor_assignment, output_dir, folder, epoch, annot=True):
    cluster_dir = os.path.join(output_dir, folder)
    os.makedirs(cluster_dir, exist_ok=True)
    neuron_factor_assignments = np.concatenate(factor_assignment, axis=0)
    plt.figure(figsize=(10, 30))
    sns.heatmap(neuron_factor_assignments, cmap="YlOrRd", annot=annot,
                cbar_kws={'label': 'Assignment probability', 'location': 'bottom'},
                vmin=0, vmax=1)
    plt.title('Neuron factor cluster assignments')
    plt.savefig(os.path.join(cluster_dir, f'cluster_assn_{epoch}.png'), dpi=200)
    plt.close()


def plot_data_dispersion(Y, factor_access, n_areas, save_path, save_folder, regions, W_CKL=None):
    # Y # K x T x R x C
    # factor_access  # C x K x L
    _, T, R, _ = Y.shape
    if W_CKL is None:
        filter = factor_access
    else:
        filter = W_CKL
    spike_counts = torch.einsum('ktrc,ckl->ckl', Y, filter)
    avg_spike_counts = torch.sum(spike_counts, dim=(0, 1)) / torch.sum(filter, dim=(0, 1))
    sq_centered_spike_counts = (spike_counts - avg_spike_counts.unsqueeze(0).unsqueeze(1))**2 * filter
    spike_ct_var = torch.sum(sq_centered_spike_counts, dim=(0,1)) / (torch.sum(filter, dim=(0, 1)))
    dispersion = (spike_ct_var/avg_spike_counts).reshape(n_areas, -1)[:, 0]
    plt.figure(figsize=(10, 6))
    plt.bar(regions, dispersion)
    plt.xlabel('Area')
    plt.ylabel('Dispersion ratio (V/mu)')
    plt.title('Dispersion ratio of Spike Counts')
    folder_name = os.path.join(save_path, save_folder)
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, 'data_dispersion_ratio.png'), dpi=200)
    plt.close()


def write_losses(list, name, metric, output_dir, starts_out_empty):
    file_name = f'{metric}_{name}.json'
    file_dir = os.path.join(output_dir, 'json', file_name)
    if not os.path.exists(file_dir):
        raise Exception(f'File {file_name} has not been created yet')
    with open(file_dir, 'r+b') as file:
        _ = file.seek(-1, 2)  # Go to the one character before the end of the file
        if file.read(1) != b']':
            raise ValueError("JSON file must end with a ']'")
        _ = file.seek(-1, 2)  # Go back to the position just before the ']'
        currently_empty = starts_out_empty
        for item in list:
            if not currently_empty:
                _ = file.write(b',' + json.dumps(item).encode('utf-8'))
            else:
                _ = file.write(json.dumps(item).encode('utf-8'))
                currently_empty = 0
        _ = file.write(b']')


def plot_losses(true_likelihood, output_dir, name, metric, cutoff=0, merge=True):
    file_name = f'{metric}_{name}.json'
    if name == 'test':
        folder = 'Test'
    else:
        folder = 'Train'
    if name == 'batch':
        epoch_file_name = 'epoch_batch.json'
    else:
        epoch_file_name = 'epoch_train.json'
    plt_path = os.path.join(output_dir, folder, 'Trajectories')
    os.makedirs(plt_path, exist_ok=True)

    path_info = output_dir.split('Run_')
    parent_dir = path_info[0]
    run_number = int(path_info[1])
    if merge:
        run_folders = sorted([folder for folder in os.listdir(parent_dir) if int(folder.split('Run_')[-1]) <= run_number])
    else:
        run_folders = [f'Run_{run_number}']

    metric_data = []
    epoch_data = []
    for run_fldr in run_folders:
        output_dir = os.path.join(parent_dir, run_fldr, 'json')
        json_path = os.path.join(output_dir, file_name)
        with open(json_path, 'r') as file:
            metric_data.extend(json.load(file))
        epoch_json_path = os.path.join(output_dir, epoch_file_name)
        with open(epoch_json_path, 'r') as file:
            epoch_data.extend(json.load(file))

    metric_data = metric_data[cutoff:]
    epoch_data = epoch_data[cutoff:]
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_data, metric_data, label=metric)
    if true_likelihood is not None:
        true_likelihood_vector = [true_likelihood] * len(metric_data)
        plt.plot(epoch_data, true_likelihood_vector, label='True Log Likelihood')
    if 'MSE' in metric:
        plt.ylim(bottom=0)
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Plot of metric values')
    plt.legend()
    if cutoff > 0:
        plt.savefig(os.path.join(plt_path, f'{metric}_{name}_Trajectories_Cutoff{cutoff}.png'))
    else:
        plt.savefig(os.path.join(plt_path, f'{metric}_{name}_Trajectories.png'))


def write_grad_norms(norms, name, output_dir, starts_out_empty):
    for param, list in norms.items():
        file_name = f'{param}_{name}_grad_norms.json'
        file_dir = os.path.join(output_dir, 'json', file_name)
        if not os.path.exists(file_dir):
            raise Exception(f'File {file_name} has not been created yet')
        with open(file_dir, 'r+b') as file:
            _ = file.seek(-1, 2)
            if file.read(1) != b']':
                raise ValueError("JSON file must end with a ']'")
            _ = file.seek(-1, 2)
            currently_empty = starts_out_empty
            for item in list:
                if not currently_empty:
                    _ = file.write(b',' + json.dumps(item).encode('utf-8'))
                else:
                    _ = file.write(json.dumps(item).encode('utf-8'))
                    currently_empty = 0
            _ = file.write(b']')


def plot_grad_norms(norms_list, output_dir, name, cutoff=0, merge=True):
    if name == 'batch':
        epoch_file_name = 'epoch_batch.json'
    else:
        epoch_file_name = 'epoch_train.json'
    plt_path = os.path.join(output_dir, 'Train', 'Trajectories')
    os.makedirs(plt_path, exist_ok=True)

    path_info = output_dir.split('Run_')
    parent_dir = path_info[0]
    run_number = int(path_info[1])
    if merge:
        run_folders = sorted([folder for folder in os.listdir(parent_dir) if int(folder.split('Run_')[-1]) <= run_number])
    else:
        run_folders = [f'Run_{run_number}']

    for param in norms_list:
        file_name = f'{param}_{name}_grad_norms.json'
        metric_data = []
        epoch_data = []
        for run_fldr in run_folders:
            output_dir = os.path.join(parent_dir, run_fldr, 'json')
            json_path = os.path.join(output_dir, file_name)
            with open(json_path, 'r') as file:
                metric_data.extend(json.load(file))
            epoch_json_path = os.path.join(output_dir, epoch_file_name)
            with open(epoch_json_path, 'r') as file:
                epoch_data.extend(json.load(file))
            metric_data = metric_data[cutoff:]
            epoch_data = epoch_data[cutoff:]
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_data, metric_data, label='grad norms')
            plt.xlabel('Iterations')
            plt.ylabel('Gradient Norm')
            plt.title(f'Plot of {param} gradient norms')
            plt.legend()
            if cutoff > 0:
                plt.savefig(os.path.join(plt_path, f'{param}_{name}_grad_norms_Trajectories_Cutoff{cutoff}.png'))
            else:
                plt.savefig(os.path.join(plt_path, f'{param}_{name}_grad_norms_Trajectories.png'))


def compute_uncertainty(model, processed_inputs, output_dir, epoch):
    # check if file exists
    models_path = os.path.join(output_dir, 'models')
    os.makedirs(models_path, exist_ok=True)
    file_path = os.path.join(models_path, f'se_dict_{epoch}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            se_dict = pickle.load(f)
        return se_dict

    likelihood_term = model.log_likelihood(processed_inputs, E_step=True)
    param_names = ['alpha', 'beta', 'config_peak_offsets', 'trial_peak_offset_covar_ltri_diag',
                   'trial_peak_offset_covar_ltri_offdiag', 'trial_peak_offset_proposal_means']
    model_named_parameters = dict(model.named_parameters())
    param_values = {n: model_named_parameters[n] for n in param_names}
    first_grads = {n: v.flatten() for n, v in zip(param_names, torch.autograd.grad(likelihood_term, list(param_values.values()), create_graph=True))}
    outputs = {}

    def compute_hessian(param_name1, param_name2):
        w_r_t1 = first_grads[param_name1]
        num_iter = len(w_r_t1)
        unit_matrix = torch.eye(num_iter, device=model.device)
        w_r_t2 = param_values[param_name2]
        print(f'Computing Hessian w.r.t {param_name1}, {param_name2}')
        second_grads = []
        for i in range(num_iter):
            second_grads.append((torch.autograd.grad(w_r_t1, w_r_t2, unit_matrix[i], retain_graph=True)[0]).flatten()[None, :])
        outputs[f'{param_name1}_{param_name2}'] = torch.cat(second_grads, dim=0)

    compute_hessian('alpha', 'alpha')
    compute_hessian('alpha', 'beta')
    compute_hessian('beta', 'beta')
    compute_hessian('trial_peak_offset_covar_ltri_diag', 'trial_peak_offset_covar_ltri_diag')
    compute_hessian('trial_peak_offset_covar_ltri_diag', 'trial_peak_offset_covar_ltri_offdiag')
    compute_hessian('trial_peak_offset_covar_ltri_offdiag', 'trial_peak_offset_covar_ltri_offdiag')

    alpha_beta_hess = torch.cat((torch.cat((outputs['alpha_alpha'], outputs['alpha_beta']), dim=1),
                                 torch.cat((outputs['alpha_beta'].t(), outputs['beta_beta']), dim=1)), dim=0)
    alpha_beta_hess = alpha_beta_hess + torch.eye(alpha_beta_hess.shape[0], device=model.device) * 1e-6

    ltri_hess = torch.cat(
        (torch.cat((outputs['trial_peak_offset_covar_ltri_diag_trial_peak_offset_covar_ltri_diag'],
                    outputs['trial_peak_offset_covar_ltri_diag_trial_peak_offset_covar_ltri_offdiag']), dim=1),
         torch.cat((outputs['trial_peak_offset_covar_ltri_diag_trial_peak_offset_covar_ltri_offdiag'].t(),
                    outputs['trial_peak_offset_covar_ltri_offdiag_trial_peak_offset_covar_ltri_offdiag']), dim=1)), dim=0)
    ltri_hess = ltri_hess + torch.eye(ltri_hess.shape[0], device=model.device) * 1e-6

    alpha_beta_inv = torch.abs(torch.round(torch.inverse(-alpha_beta_hess).diag(), decimals=2))
    ltri_inv = torch.abs(torch.round(torch.inverse(-ltri_hess).diag(), decimals=2))

    alpha_se = torch.sqrt(alpha_beta_inv[:param_values['alpha'].shape[0]]) #  .numpy()[:,None]
    beta_se = torch.sqrt(alpha_beta_inv[param_values['alpha'].shape[0]:]).reshape(*param_values['beta'].shape) #  .numpy().T
    ltri_diag_se = torch.sqrt(ltri_inv[:param_values['trial_peak_offset_covar_ltri_diag'].shape[0]]) #  .numpy()[:,None]
    ltri_offdiag_se = torch.sqrt(ltri_inv[param_values['trial_peak_offset_covar_ltri_diag'].shape[0]:]) #  .numpy()[:,None]

    se_dict = {'alpha': alpha_se.cpu(), 'beta': beta_se.cpu(), 'ltri_diag': ltri_diag_se.cpu(), 'ltri_offdiag': ltri_offdiag_se.cpu()}
    with open(file_path, 'wb') as f:
        pickle.dump(se_dict, f)
    return se_dict


def compute_warped_factors(model, data, warped_times):
    factors = torch.einsum('ktrc,ckl->lcrt', data, model.W_CKL)
    r0und = model.dt / (10 * int(str(model.dt.item())[-1]))
    warped_indices = (warped_times / model.dt) + r0und  # could be round but its not differentiable
    floor_warped_indices = warped_indices.int()  # could be torch.floor but for non-negative numbers it is the same
    ceil_warped_indices = (warped_indices + 1).int()  # could be torch.ceil but for non-negative numbers it is the same
    ceil_weights = warped_indices - floor_warped_indices
    floor_weights = 1 - ceil_weights
    full_warped_factors = []
    L, C, R, T = factors.shape
    for l in range(L):
        full_warped_factors_l = []
        for c in range(C):
            full_warped_factors_c = []
            for r in range(R):
                floor_warped_factor_l = factors[l, c, r, floor_warped_indices[l, :, 0, r, c]]
                weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[l, :, 0, r, c]
                ceil_warped_factor_l = factors[l, c, r, ceil_warped_indices[l, :, 0, r, c]]
                weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[l, :, 0, r, c]
                peak1 = weighted_floor_warped_factor_l + weighted_ceil_warped_factor_l

                floor_warped_factor_l = factors[l, c, r, floor_warped_indices[l + model.n_factors, :, 0, r, c]]
                weighted_floor_warped_factor_l = floor_warped_factor_l * floor_weights[l + model.n_factors, :, 0, r, c]
                ceil_warped_factor_l = factors[l, c, r, ceil_warped_indices[l + model.n_factors, :, 0, r, c]]
                weighted_ceil_warped_factor_l = ceil_warped_factor_l * ceil_weights[l + model.n_factors, :, 0, r, c]
                peak2 = weighted_floor_warped_factor_l + weighted_ceil_warped_factor_l

                early = factors[l, c, r, :model.left_landmarks_indx[l]]
                peak1 = peak1[:model.landmark_indx_speads[l]]
                mid = factors[l, c, r, (model.right_landmarks_indx[l]+1):model.left_landmarks_indx[l + model.n_factors]]
                peak2 = peak2[:model.landmark_indx_speads[l + model.n_factors]]
                late = factors[l, c, r, (model.right_landmarks_indx[l + model.n_factors]+1):]
                full_warped_factors_c.append(torch.cat([early, peak1, mid, peak2, late], dim=0))
            full_warped_factors_l.append(torch.stack(full_warped_factors_c))
        full_warped_factors.append(torch.stack(full_warped_factors_l))
    full_warped_factors = torch.stack(full_warped_factors)
    # full_warped_factors  # AL x C x R x T
    return full_warped_factors


def reverse_warp_data(model, Y):
    avg_peak_times, left_landmarks, right_landmarks, s_new = model.compute_offsets_and_landmarks()
    warped_times = model.compute_warped_times(s_new, left_landmarks, right_landmarks, avg_peak_times.expand_as(s_new))
    warped_factors = compute_warped_factors(model, Y, warped_times)
    return warped_factors


def interpret_results(model, unique_regions, factors_of_interest, output_dir, epoch):
    with torch.no_grad():
        # Interpret trial
        factors_of_interest = torch.tensor(factors_of_interest) - 1
        avg_peak_times, _, _, peak_times = model.compute_offsets_and_landmarks()
        avg_peak_times = avg_peak_times.squeeze()
        peak_times = peak_times.squeeze()
        # peak time # 2 x A x L x C x R
        peak_times = peak_times.reshape(*peak_times.shape[:2], 2, model.n_areas, -1).permute(2, 3, 4, 1, 0)
        # avg_peak_times # 2 x AL
        avg_peak_times = avg_peak_times.reshape(2, model.n_areas, -1)
        indices_of_interest = torch.zeros_like(peak_times)
        avg_indices_of_interest = torch.zeros_like(avg_peak_times)
        # Interpret factors of interest
        for i, x in enumerate(factors_of_interest):
            indices_of_interest[:, i, x] = 1
            avg_indices_of_interest[:, i, x] = 1
        # peak time of interest # 2A x C x R
        # :A is peak 1, A: is peak 2
        dimns = 2 * model.n_areas
        peak_times_of_interest = peak_times[indices_of_interest.bool()].reshape(dimns, *peak_times.shape[3:])
        avg_peak_times_of_interest = avg_peak_times[avg_indices_of_interest.bool()]
        ltri_matix = model.ltri_matix()
        inv_sigma = ltri_matix @ ltri_matix.t()
        # sigma # 2 x A x L x 2 x A x L
        L = model.n_factors // model.n_areas
        inv_sigma = inv_sigma.reshape(2, model.n_areas, L, 2, model.n_areas, L)
        indices_of_interest = torch.zeros_like(inv_sigma)
        for i, x in enumerate(factors_of_interest):
            indices_of_interest[:, i, x, :, range(len(factors_of_interest)), factors_of_interest] = 1
        # sigma of interest # 2A x 2A
        # :A is peak 1, A: is peak 2
        precision_of_interest = inv_sigma[indices_of_interest.bool()].reshape(dimns, dimns)

    inference_dir = os.path.join(output_dir, 'interpretations')
    os.makedirs(inference_dir, exist_ok=True)
    # plot average peak times
    avg_peak_df = pd.DataFrame(avg_peak_times_of_interest.reshape(2, -1).t(), columns=['Peak 1', 'Peak 2'], index=unique_regions).sort_values(by='Peak 1')
    scale = 1000
    avg_peak_df = avg_peak_df * scale
    colors = cm.get_cmap('tab10', len(avg_peak_df.index))
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(avg_peak_df.columns):
        upper_limit = avg_peak_df[col].max()
        lower_limit = avg_peak_df[col].min()
        for j, (idx, row) in enumerate(avg_peak_df.iterrows()):
            color = colors(j)
            plt.subplot(1, 2, i + 1)
            plt.text(row[col], 0, f'{idx}', ha='center', va='bottom', bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))
            plt.yticks([])
            plt.xlim(left=lower_limit - (1/50)*lower_limit, right=upper_limit + (1/50)*lower_limit)
            plt.ylim(bottom=-0.01, top=0.01)
            if i == 0:
                plt.ylabel('Region')
            plt.xlabel("Peak Time (ms)")
            plt.title(f'Average Peak {i + 1} Times')
            plt.tight_layout()
    plt.savefig(os.path.join(inference_dir, f'Average_Peak_Times_{epoch}.png'))
    plt.close()

    peak_times_of_interest = peak_times_of_interest.permute(1, 0, 2)
    colors = cm.get_cmap('tab20b', len(unique_regions))
    plt.figure(figsize=(40, 15))
    for c in range(model.n_configs):
        plt.subplot(5, 8, c + 1)
        for a in range(2 * model.n_areas):
            if (peak_times_of_interest[c, a].round(decimals=4) / peak_times_of_interest[c, a].round(decimals=4)[0]).sum().item() == peak_times_of_interest.shape[-1]:
                continue
            for t in range(model.n_trials):
                if c == 0 and t == 0 and a // model.n_areas == 0:
                    plt.plot(peak_times_of_interest[c, a, t], t, '.', color=colors(a % model.n_areas), label=unique_regions[a % model.n_areas])
                else:
                    plt.plot(peak_times_of_interest[c, a, t], t, '.', color=colors(a % model.n_areas))
        if c == 0:
            plt.legend()
        plt.xlabel('Peak Time (ms)')
        plt.ylabel('Config Trial')
        plt.tight_layout()
    plt.savefig(os.path.join(inference_dir, f'Trial_Peak_Times_{epoch}.png'))
    plt.close()

    partial_correlation = -precision_of_interest / np.sqrt(np.outer(np.diag(precision_of_interest), np.diag(precision_of_interest)))
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(partial_correlation, annot=True, cmap="seismic", center=0, vmin=-1, vmax=1,
                     xticklabels=model.n_areas, yticklabels=model.n_areas)
    for i in range(model.n_areas, partial_correlation.shape[0], model.n_areas):
        ax.axvline(i, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(i, color='black', linestyle='-', linewidth=0.5)
    plt.title('Peak time correlation matrix')
    plt.savefig(os.path.join(inference_dir, f'Peak_Times_Correlations_{epoch}.png'))
    plt.close()


def plot_epoch_results(input_dict, test=False):
    plot_training_epoch_results(input_dict)
    if test:
        plot_test_epoch_results(input_dict)


def plot_training_epoch_results(input_dict):
    print('Plotting epoch results')
    Y, factor_access, unique_regions, output_dir, epoch = input_dict['Y'], input_dict['neuron_factor_access'], input_dict['unique_regions'], input_dict['output_dir'], input_dict['epoch']
    batch_grad_norms, grad_norms = input_dict['batch_grad_norms'], input_dict['grad_norms']
    model_state, optimizer_state, scheduler_state, W_CKL, a_CKL, theta, pi, epoch = load_model_checkpoint(output_dir, epoch)
    model = LikelihoodELBOModel(**input_dict['model_params'])
    model.init_zero()
    model.load_state_dict(model_state)
    model.W_CKL, model.a_CKL, model.theta, model.pi = W_CKL, a_CKL, theta, pi
    likelihood_ground_truth_train, true_ELBO_train = input_dict['likelihood_ground_truth_train'], input_dict['true_ELBO_train']
    plot_outputs(model, unique_regions, output_dir, 'Train', epoch, Y=Y.to(model.W_CKL.dtype), factor_access=factor_access)
    plot_grad_norms(batch_grad_norms, output_dir, 'batch', 20, False)
    plot_grad_norms(grad_norms, output_dir, 'train', 10)
    plot_losses(likelihood_ground_truth_train, output_dir, 'train', 'true_log_likelihoods', 10)
    plot_losses(true_ELBO_train, output_dir, 'train', 'log_likelihoods', 10)
    plot_losses(None, output_dir, 'train', 'losses', 10)
    plot_losses(None, output_dir, 'batch', 'log_likelihoods', 20, False)
    plot_losses(None, output_dir, 'batch', 'losses', 20, False)


def plot_test_epoch_results(input_dict):
    print('Plotting epoch results')
    output_dir = input_dict['output_dir']
    true_offset_penalty_train = input_dict['true_offset_penalty_train']
    plot_losses(None, output_dir, 'test', 'beta_MSE')
    plot_losses(None, output_dir, 'test', 'alpha_MSE')
    plot_losses(None, output_dir, 'test', 'theta_MSE')
    plot_losses(None, output_dir, 'test', 'pi_MSE')
    plot_losses(None, output_dir, 'test', 'configoffset_MSE')
    plot_losses(None, output_dir, 'test', 'ltri_MSE')
    plot_losses(None, output_dir, 'test', 'Sigma_MSE')
    plot_losses(None, output_dir, 'test', 'proposal_means_MSE')
    plot_losses(None, output_dir, 'train', 'gains_MSE')
    plot_losses(true_offset_penalty_train, output_dir, 'train', 'ltriLkhd', 10)
    # true_ELBO_test, true_offset_penalty_test = input_dict['true_ELBO_test'], input_dict['true_offset_penalty_test']
    # plot_losses(true_ELBO_test, output_dir, 'test', 'log_likelihoods', 10)
    # plot_losses(None, output_dir, 'test', 'losses', 10)
    # plot_losses(true_offset_penalty_test, output_dir, 'test', 'ltriLkhd', 10)
    # plot_losses(None, output_dir, 'test', 'gains_MSE')


def load_tensors(arrays):
    return tuple([torch.tensor(array, dtype=torch.float64) for array in arrays])


def to_cuda(tensors, move_to_cuda=True):
    if not move_to_cuda:
        return tensors
    return tuple([tensor.cuda() for tensor in tensors])


def to_cpu(tensors):
    return tuple([tensor.cpu() for tensor in tensors])


def softplus(x):
    return np.log(1 + np.exp(x))


def inv_softplus(x, threshold=20):
    return np.where(x < threshold, np.log(np.exp(x) - 1), x)


def inv_softplus_torch(x, threshold=20):
    return torch.where(x < threshold, torch.log(torch.exp(x) - 1), x)


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


def remove_minimums(true_likelihood, output_dir, name, metric, min_cutoff=0, write=False, write_cutoff=0):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    elif 'loss' in metric.lower():
        file_name = 'losses'
    else:
        file_name = metric
    file_name = f'{file_name}_{name.lower()}.json'
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, 'r') as file:
        metric_data = json.load(file)

    metric_data_plot = metric_data[min_cutoff:]
    min_indx = np.min(np.where(metric_data_plot==np.min(metric_data_plot))) + min_cutoff
    metric_data_plot = metric_data
    metric_data_plot[min_indx] = metric_data_plot[min_indx + 1]

    plt.figure(figsize=(10, 6))
    plt.plot(metric_data_plot, label=metric)
    if 'likelihood' in metric.lower():
        true_likelihood_vector = [true_likelihood] * len(metric_data_plot)
        plt.plot(true_likelihood_vector, label='True Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Plot of metric values')
    plt.legend()
    plt.show()
    if min_indx != 0:
        print(f'Found minimum at index {min_indx}')
        if write:
            metric_data[min_indx] = metric_data[min_indx + 1]
            with open(json_path, 'w') as file:
                json.dump(metric_data, file)
            plot_losses(true_likelihood, output_dir, name, metric, write_cutoff)
            print(f'Removed minimum at index {min_indx}')
    else:
        print(f'No minimum found')


def remove_chunk(true_likelihood, output_dir, name, metric, start, end, write=False, cutoff=0):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    elif 'loss' in metric.lower():
        file_name = 'losses'
    else:
        file_name = metric
    file_name = f'{file_name}_{name.lower()}.json'
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, 'r') as file:
        metric_data = json.load(file)
    metric_data_plot = metric_data[cutoff:]
    if end == -1:
        end = len(metric_data_plot)
    metric_data1 = metric_data_plot[:start]
    metric_data2 = metric_data_plot[end:]
    metric_data_plot = metric_data1 + metric_data2

    plt.figure(figsize=(10, 6))
    plt.plot(metric_data_plot, label=metric)
    if 'likelihood' in metric.lower():
        true_likelihood_vector = [true_likelihood] * len(metric_data_plot)
        plt.plot(true_likelihood_vector, label='True Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Plot of metric values')
    plt.legend()
    plt.show()
    if write:
        metric_data = metric_data_plot
        with open(json_path, 'w') as file:
            json.dump(metric_data, file)
        plot_losses(true_likelihood, output_dir, name, metric, cutoff)
