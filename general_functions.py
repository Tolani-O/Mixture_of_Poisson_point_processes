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
    parser.add_argument('--tau_config', type=int, default=0.5, help='Value for tau_sigma1')
    parser.add_argument('--tau_sigma', type=int, default=0.5, help='Value for tau_sigma2')
    parser.add_argument('--tau_beta', type=int, default=0.5, help='Value for tau_beta')
    parser.add_argument('--tau_budget', type=int, default=0.5, help='Value for tau_tau_budget')
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--scheduler_patience', type=int, default=1000, help='Number of epochs before scheduler step')
    parser.add_argument('--scheduler_factor', type=int, default=0.8, help='Value for tau_tau_budget')
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


def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # set the last element to 1
    D[-1, -1] = 1
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
    load_optimizer_dir = os.path.join(output_dir, 'models', f'optimizer_{load_epoch}.pth')
    scheduler_dir = os.path.join(output_dir, 'models', f'scheduler_{load_epoch}.pth')
    if os.path.isfile(load_model_dir):
        model = torch.load(load_model_dir)
    else:
        raise Exception(f'No model_{load_epoch}.pth file found at {load_model_dir}')
    if os.path.isfile(load_optimizer_dir):
        optimizer = torch.load(load_optimizer_dir)
    else:
        raise Exception(f'No optimizer_{load_epoch}.pth file found at {load_optimizer_dir}')
    if os.path.isfile(scheduler_dir):
        scheduler = torch.load(scheduler_dir)
    else:
        raise Exception(f'No scheduler_{load_epoch}.pth file found at {scheduler_dir}')
    return model, optimizer, scheduler


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


def create_relevant_files(output_dir, output_str):
    with open(os.path.join(output_dir, 'log.txt'), 'w') as file:
        file.write(output_str)

    # List of JSON files to be created
    file_names = [
        'log_likelihoods_batch',
        'losses_batch',
        'log_likelihoods_train',
        'losses_train',
        'log_likelihoods_test',
        'losses_test',
        'beta_MSE_test',
        'alpha_MSE_test',
        'theta_MSE_test',
        'pi_MSE_test',
        'configoffset_MSE_test',
        'ltri_MSE_test',
        'clusr_misses_train',
        'clusr_misses_test',
        'gains_MSE_train',
        'gains_MSE_test',
        'trialoffsets_MSE_train',
        'trialoffsets_MSE_test',
    ]

    # Iterate over the filenames and create each file
    for file_name in file_names:
        with open(os.path.join(output_dir, '.'.join([file_name, 'json'])), 'w+b') as file:
            file.write(b'[]')

    # command_str = (f"python src/psplines_gradient_method/main.py "
    #                f"--K {args.K} --R {args.n_trials} --L {args.L} --intensity_mltply {args.intensity_mltply} "
    #                f"--intensity_bias {args.intensity_bias} --tau_beta {args.tau_beta} --tau_sigma1 {args.tau_sigma1} "
    #                f"--num_epochs {args.num_epochs} --notes {args.notes} "
    #                f"--data_seed {args.data_seed} --param_seed {args.param_seed} --load_and_train 1")
    # with open(os.path.join(output_dir, 'command.txt'), 'w') as file:
    #     file.write(command_str)


def write_log_and_model(output_str, output_dir, epoch, model, optimizer, scheduler):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
        file.write(output_str)
    models_path = os.path.join(output_dir, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(model.state_dict(), os.path.join(models_path, f'model_{epoch}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(models_path, f'optimizer_{epoch}.pth'))
    torch.save(scheduler.state_dict(), os.path.join(models_path, f'scheduler_{epoch}.pth'))


def plot_outputs(model, n_areas, output_dir, folder, epoch):

    output_dir = os.path.join(output_dir, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    beta_dir = os.path.join(output_dir, 'beta')
    if not os.path.exists(beta_dir):
        os.makedirs(beta_dir)
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
    with torch.no_grad():
        latent_factors = torch.exp(model.beta).numpy()
        L = latent_factors.shape[0]
        global_max = np.max(latent_factors)
        upper_limit = global_max + 0.01
        plt.figure(figsize=(10, L*5))
        for l in range(L):
            plt.subplot(L, 1, l + 1)
            plt.plot(latent_factors[l, :], label=f'Factor [{l}, :]')
            plt.title(f'Factor [{l}, :]')
            plt.ylim(bottom=0, top=upper_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(beta_dir, f'LatentFactors_{epoch}.png'))

        alpha = F.softplus(model.alpha).numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(alpha, label='Alpha')
        plt.title('Alpha')
        plt.savefig(os.path.join(alpha_dir, f'alpha_{epoch}.png'))

        theta = F.softplus(model.theta).numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(theta, label='Theta')
        plt.title('Theta')
        plt.savefig(os.path.join(theta_dir, f'theta_{epoch}.png'))

        pi = F.softmax(torch.cat([torch.zeros(n_areas, 1), model.pi], dim=1), dim=1).flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(pi, label='Pi')
        plt.title('Pi')
        plt.savefig(os.path.join(pi_dir, f'pi_{epoch}.png'))

        configoffset = model.config_peak_offsets.flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(configoffset, label='ConfigOffset')
        plt.title('ConfigOffset')
        plt.savefig(os.path.join(configoffset_dir, f'configoffset_{epoch}.png'))

        ltri = model.trial_peak_offset_covar_ltri.flatten().numpy()
        plt.figure(figsize=(10, 10))
        plt.plot(ltri, label='Ltri')
        plt.title('Ltri')
        plt.savefig(os.path.join(ltri_dir, f'ltri_{epoch}.png'))


    # warped_intensities = warped_factors.reshape(-1, 200)
    #
    # global_max = np.max(warped_intensities)
    # upper_limit = global_max + batch * 0.01
    # num_of_warped = warped_intensities.shape[0]
    # for i in range(0, num_of_warped, batch):
    #     this_batch = batch if i + batch < num_of_warped else num_of_warped - i
    #     plt.figure(figsize=(10, 10))
    #     for j in range(this_batch):
    #         plt.plot(stim_time, warped_intensities[i + j, :] + j * 0.01)
    #         plt.ylim(bottom=0, top=upper_limit)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, f'warped_intensities_batch{i}.png'))


def plot_factor_assignments(factor_assignment, output_dir, folder, epoch):
    cluster_dir = os.path.join(output_dir, folder, 'Cluster')
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    neuron_factor_assignments = np.concatenate(factor_assignment, axis=0)
    height, width = [a / b * 10 for a, b in zip(neuron_factor_assignments.shape, (5, 1))]
    plt.figure(figsize=(20, 60))
    sns.heatmap(neuron_factor_assignments, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Neuron factor cluster assignments')
    plt.savefig(os.path.join(cluster_dir,  f'cluster_assn_{epoch}.png'))


def write_losses(list, name, metric, output_dir, starts_out_empty):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    elif 'loss' in metric.lower():
        file_name = 'losses'
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
    else:
        file_name = metric
    file_name = f'{file_name}_{name.lower()}.json'
    if name.lower()=='test':
        folder = 'Test'
    else:
        folder = 'Train'
    plt_path = os.path.join(output_dir, folder)
    if not os.path.exists(plt_path):
        os.makedirs(plt_path)
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, 'r') as file:
        metric_data = json.load(file)
    metric_data = metric_data[cutoff:]
    plt.figure(figsize=(10, 6))
    plt.plot(metric_data, label=metric)
    if 'likelihood' in metric.lower():
        true_likelihood_vector = [true_likelihood] * len(metric_data)
        plt.plot(true_likelihood_vector, label='True Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Plot of metric values')
    plt.legend()
    if cutoff > 0:
        plt.savefig(os.path.join(plt_path, f'{metric}_{name}_Trajectories_Cutoff{cutoff}.png'))
    else:
        plt.savefig(os.path.join(plt_path, f'{metric}_{name}_Trajectories.png'))


def load_tensors(numpys, is_cuda):
    tensors = [torch.tensor(numpy) for numpy in numpys]
    if is_cuda:
        return tuple([tensor.cuda() for tensor in tensors])
    else:
        return tensors


def softplus(x):
    return np.log(1 + np.exp(x))


def inv_softplus(x):
    return np.log(np.exp(x) - 1)


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
