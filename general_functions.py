import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.figure import figaspect
import json
import argparse
from torch.utils.data import Dataset
from tslearn.clustering import TimeSeriesKMeans
from scipy.ndimage import gaussian_filter1d
import pickle
sns.set()
plt.rcParams.update({'figure.max_open_warning': 0})

def get_parser():
    parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA (default: True)')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials per stimulus condition')
    parser.add_argument('--n_configs', type=int, default=2, help='Number of stimulus conditions')
    parser.add_argument('--A', type=int, default=2, help='Number of areas')
    parser.add_argument('--n_trial_samples', type=int, default=10, help='Number of trial samples for monte carlo integration')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
    parser.add_argument('--load', type=int, default=0, help='')
    parser.add_argument('--load_epoch', type=int, default=0, help='Which epoch to load model and optimizer from')
    parser.add_argument('--load_run', type=int, default=0, help='Which run to load model and optimizer from')
    parser.add_argument('--tau_config', type=float, default=0.5, help='Value for tau_config')
    parser.add_argument('--tau_sigma', type=float, default=0.5, help='Value for tau_sigma')
    parser.add_argument('--tau_sd', type=float, default=0.5, help='Value for tau_sd')
    parser.add_argument('--tau_beta', type=float, default=0.5, help='Value for tau_beta')
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--scheduler_patience', type=int, default=1000, help='Number of epochs before scheduler step')
    parser.add_argument('--scheduler_factor', type=int, default=0.8, help='Scheduler reduction factor')
    parser.add_argument('--scheduler_threshold', type=int, default=10, help='Threshold to accept step improvement')
    parser.add_argument('--notes', type=str, default='empty', help='Run notes')
    parser.add_argument('--K', type=int, default=30, help='Number of neurons')
    parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
    parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
    parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
    parser.add_argument('--param_seed', type=int_or_str, default='', help='options are: seed (int), Truth (str)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval (default: 100')
    parser.add_argument('--eval_interval', type=int, default=100, metavar='N', help='report interval (default: 10')
    parser.add_argument('--batch_size', type=int_or_str, default=5, help='the batch size for training')

    parser.add_argument('--nhid', type=int, default=150, help='number of hidden units per layer (default: 150)')
    parser.add_argument('--plot_lkhd', type=int, default=0, help='')
    parser.add_argument('--init_map_load_epoch', type=int, default=1500, help='Which epoch to load for init map')
    parser.add_argument('--init_load_folder', type=str, default='', help='Which folder to load inits from')
    parser.add_argument('--init_load_subfolder_outputs', type=str, default='', help='Which subfolder to load outputs from')
    parser.add_argument('--init_load_subfolder_map', type=str, default='', help='Which subfolder to load map from')
    parser.add_argument('--init_load_subfolder_finetune', type=str, default='', help='Which subfolder to load finetune from')
    parser.add_argument('--train', type=int, default=0, help='')
    parser.add_argument('--reset_checkpoint', type=int, default=0, help='')
    parser.add_argument('--stage', type=str, default='finetune', help='options are: initialize_output, initialize_map, finetune, endtoend')
    return parser


class CustomDataset(Dataset):
    def __init__(self, Y, neuron_factor_access):
        # Y # K x T x R x C
        # neuron_factor_access  #  K x L x C
        self.Y = Y
        self.neuron_factor_access = neuron_factor_access

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.Y[idx], self.neuron_factor_access[idx]


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
    D = np.zeros((P-1, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 1:], 1)
    return D


def create_second_diff_matrix(P, dt):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    D = D/(dt**2)
    # first row is a forward difference
    s0 = [2, -5, 4, -1]
    D0 = np.concatenate((s0, np.zeros(P-4)))/(dt**3)
    D = np.vstack((D0, D, np.flip(D0)))
    return D


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
            W_CKL, a_CKL = intermediate_vars['W_CKL'], intermediate_vars['a_CKL']
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
    return model, optimizer, scheduler, W_CKL, a_CKL


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


def create_relevant_files(output_dir, output_str, ground_truth=False):
    with open(os.path.join(output_dir, 'log.txt'), 'w') as file:
        file.write(output_str)

    # List of JSON files to be created
    file_names = [
        'log_likelihoods_batch',
        'losses_batch',
        'epoch_batch',
        'log_likelihoods_train',
        'losses_train',
        'epoch_train',

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
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(model.state_dict(), os.path.join(models_path, f'model_{epoch}.pth'))
    with open(os.path.join(models_path, f'intermediate_vars_{epoch}.pkl'), 'wb') as f:
        pickle.dump({'W_CKL': model.W_CKL, 'a_CKL': model.a_CKL}, f)
    torch.save(optimizer.state_dict(), os.path.join(models_path, f'optimizer_{epoch}.pth'))
    torch.save(scheduler.state_dict(), os.path.join(models_path, f'scheduler_{epoch}.pth'))


def plot_outputs(model, neuron_factor_access, unique_regions, output_dir, folder, epoch):

    output_dir = os.path.join(output_dir, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    beta_dir = os.path.join(output_dir, 'beta')
    if not os.path.exists(beta_dir):
        os.makedirs(beta_dir)
    log_beta_dir = os.path.join(output_dir, 'log_beta')
    if not os.path.exists(log_beta_dir):
        os.makedirs(log_beta_dir)
    alpha_dir = os.path.join(output_dir, 'alpha')
    if not os.path.exists(alpha_dir):
        os.makedirs(alpha_dir)
    theta_dir = os.path.join(output_dir, 'theta')
    if not os.path.exists(theta_dir):
        os.makedirs(theta_dir)
    pi_dir = os.path.join(output_dir, 'pi')
    if not os.path.exists(pi_dir):
        os.makedirs(pi_dir)
    configoffset_dir = os.path.join(output_dir, 'configoffset')
    if not os.path.exists(configoffset_dir):
        os.makedirs(configoffset_dir)
    ltri_dir = os.path.join(output_dir, 'ltri')
    if not os.path.exists(ltri_dir):
        os.makedirs(ltri_dir)
    sigma_dir = os.path.join(output_dir, 'sigma')
    if not os.path.exists(sigma_dir):
        os.makedirs(sigma_dir)
    proposal_means_dir = os.path.join(output_dir, 'proposal_means')
    if not os.path.exists(proposal_means_dir):
        os.makedirs(proposal_means_dir)
    trial_sd_dir = os.path.join(output_dir, 'trial_SDs')
    if not os.path.exists(trial_sd_dir):
        os.makedirs(trial_sd_dir)
    with torch.no_grad():
        beta = model.unnormalized_log_factors().numpy()
        L = beta.shape[0]
        global_max = np.max(beta)
        upper_limit = global_max + 0.1
        global_min = np.min(beta)
        lower_limit = global_min - 0.01
        plt.figure(figsize=(10, L*5))
        for l in range(L):
            plt.subplot(L, 1, l + 1)
            plt.plot(beta[l, :], label=f'Log Factor [{l}, :]')
            plt.title(f'Log Factor [{l}, :]')
            plt.ylim(bottom=lower_limit, top=upper_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(log_beta_dir, f'beta_{epoch}.png'))
        plt.close()

        pi = model.pi_value(neuron_factor_access).numpy()
        if model.W_CKL is None:
            W_L = np.zeros(model.n_factors)
        else:
            plot_factor_assignments(model.W_CKL.numpy(), output_dir, 'cluster', epoch, False)
            W_L = np.round((torch.sum(model.W_CKL, dim=(0, 1))/model.W_CKL.shape[0]).numpy(), 1)
        latent_factors = torch.softmax(model.unnormalized_log_factors(), dim=-1).numpy()
        global_max = np.max(latent_factors)
        upper_limit = global_max + 0.005
        plt.figure(figsize=(model.n_areas*10, int(model.n_factors/model.n_areas)*5))
        L = np.arange(model.n_factors).reshape(model.n_areas, -1).T.flatten()
        c = 0
        factors_per_area = int(model.n_factors/model.n_areas)
        for l in L:
            plt.subplot(factors_per_area, model.n_areas, c + 1)
            plt.plot(model.time, latent_factors[l, :], label=f'Factor [{l}, :]', alpha=np.max([pi[l], 0.7]), linewidth=np.exp(2.5*pi[l]))
            plt.vlines(x=model.time[torch.tensor([
                model.peak1_left_landmarks[l], model.peak1_right_landmarks[l],
                model.peak2_left_landmarks[l], model.peak2_right_landmarks[l]])],
                       ymin=0, ymax=upper_limit,
                       color='grey', linestyle='--', alpha=0.5)
            plt.title(f'Factor {(l%factors_per_area)+1}, '
                      f'Area {unique_regions[l//factors_per_area]}, '
                      f'Membership: {pi[l]:.2f}, '
                      f'Count: {W_L[l]:.1f}', fontsize=20)
            plt.ylim(bottom=0, top=upper_limit)
            c += 1
        plt.tight_layout()
        plt.savefig(os.path.join(beta_dir, f'LatentFactors_{epoch}.png'))
        plt.close()

        alpha = F.softplus(model.alpha).numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(alpha, label='Alpha')
        plt.title('Alpha')
        plt.savefig(os.path.join(alpha_dir, f'alpha_{epoch}.png'))
        plt.close()

        theta = model.theta_value().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(theta, label='Theta')
        plt.title('Theta')
        plt.savefig(os.path.join(theta_dir, f'theta_{epoch}.png'))
        plt.close()

        pi = model.pi_value(neuron_factor_access).numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(pi, label='Pi')
        plt.title('Pi')
        plt.savefig(os.path.join(pi_dir, f'pi_{epoch}.png'))
        plt.close()

        configoffset = model.config_peak_offsets.flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(configoffset, label='ConfigOffset')
        plt.title('ConfigOffset')
        plt.savefig(os.path.join(configoffset_dir, f'configoffset_{epoch}.png'))
        plt.close()

        ltri = model.ltri_matix('cpu').flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(ltri, label='Ltri')
        plt.title('Ltri')
        plt.savefig(os.path.join(ltri_dir, f'ltri_{epoch}.png'))
        plt.close()

        Sigma = (model.ltri_matix('cpu') @ model.ltri_matix('cpu').t()).flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(Sigma, label='Sigma')
        plt.title('Sigma')
        plt.savefig(os.path.join(sigma_dir, f'Sigma_{epoch}.png'))

        proposal_offsets = model.trial_peak_offset_proposal_means.flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(proposal_offsets, label='Trial means Proposals')
        plt.title('Trial means Proposals')
        plt.savefig(os.path.join(proposal_means_dir, f'proposal_means_{epoch}.png'))
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
    # factor_access  # K x L x C
    K, T, R, C = Y.shape
    Y_train = gaussian_filter1d(Y.sum(axis=(2, 3)), sigma=bandwidth, axis=0)
    dba_km = TimeSeriesKMeans(n_clusters=n_clusters, n_init=10, metric='dtw', max_iter_barycenter=20, n_jobs=n_jobs)
    print('Fitting DBA-KMeans')
    y_pred = dba_km.fit_predict(Y_train)
    neuron_factor_assignment = torch.zeros((K, C, n_clusters), dtype=torch.float64)
    neuron_factor_assignment[torch.arange(K), :, y_pred] = 1
    neuron_factor_assignment = torch.concat([neuron_factor_assignment] * n_areas, dim=-1).permute(1, 0, 2) * factor_access.permute(2, 0, 1)
    beta = torch.log(torch.concat([torch.tensor(dba_km.cluster_centers_).squeeze()] * n_areas, dim=0))
    # save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, 'cluster_initialization.pkl')
    print('Saving clusters to: ', save_dir)
    with open(save_dir, 'wb') as f:
        pickle.dump({'y_pred': y_pred, 'neuron_factor_assignment': neuron_factor_assignment, 'beta': beta}, f)


def plot_initial_clusters(output_dir, data_folder, n_clusters):
    # Y # K x T x R x C
    data_dir = os.path.join(output_dir, data_folder, f'{data_folder}.pkl')
    cluster_dir = os.path.join(output_dir, data_folder, 'cluster_initialization.pkl')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(cluster_dir):
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    Y, time = data['Y'], data['time']
    with open(cluster_dir, 'rb') as f:
        data = pickle.load(f)
    y_pred, neuron_factor_assignment, beta = data['y_pred'], data['neuron_factor_assignment'], data['beta']
    factors = torch.exp(beta)
    Y_train = Y.sum(axis=(2, 3))
    y_upper = torch.max(factors).item()
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        for xx in Y_train[y_pred == yi]:
            plt.plot(time, xx.ravel(), "k-", alpha=0.2)
        plt.plot(time, factors[yi], "r-")
        plt.ylim(-1, y_upper)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, data_folder, 'cluster_initialization.png'))
    plt.close()


def plot_factor_assignments(factor_assignment, output_dir, folder, epoch, annot=True):
    cluster_dir = os.path.join(output_dir, folder)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    neuron_factor_assignments = np.concatenate(factor_assignment, axis=0)
    plt.figure(figsize=(10, 30))
    sns.heatmap(neuron_factor_assignments, cmap="YlOrRd", annot=annot,
                cbar_kws={'label': 'Assignment probability', 'location': 'bottom'},
                vmin=0, vmax=1)
    plt.title('Neuron factor cluster assignments')
    plt.savefig(os.path.join(cluster_dir, f'cluster_assn_{epoch}.png'), dpi=200)
    plt.close()


def write_losses(list, name, metric, output_dir, starts_out_empty):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    elif 'loss' in metric.lower():
        file_name = 'losses'
    elif 'epoch' in metric.lower():
        file_name = 'epoch'
    else:
        file_name = metric
    file_name = f'{file_name}_{name.lower()}.json'
    file_dir = os.path.join(output_dir, file_name)
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


def plot_losses(true_likelihood, output_dir, name, metric, cutoff=0):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    elif 'loss' in metric.lower():
        file_name = 'losses'
    elif 'epoch' in metric.lower():
        file_name = 'epoch'
    else:
        file_name = metric
    file_name = f'{file_name}_{name.lower()}.json'
    if name.lower()=='test':
        folder = 'Test'
    else:
        folder = 'Train'
    if name.lower()=='batch':
        epoch_file_name = 'epoch_batch.json'
    else:
        epoch_file_name = 'epoch_train.json'
    plt_path = os.path.join(output_dir, folder)
    if not os.path.exists(plt_path):
        os.makedirs(plt_path)
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, 'r') as file:
        metric_data = json.load(file)
    epoch_json_path = os.path.join(output_dir, epoch_file_name)
    with open(epoch_json_path, 'r') as file:
        epoch_data = json.load(file)
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


def load_tensors(arrays, is_numpy=False, to_cuda=False):
    if is_numpy:
        tensors = [torch.tensor(array, dtype=torch.float64) for array in arrays]
    else:
        tensors = arrays
    if to_cuda:
        return tuple([tensor.cuda() for tensor in tensors])
    else:
        return tuple([tensor.cpu() for tensor in tensors])


def softplus(x):
    return np.log(1 + np.exp(x))


def inv_softplus(x):
    return np.log(np.exp(x) - 1)


def inv_softplus_torch(x):
    return torch.log(torch.exp(x) - 1)


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
