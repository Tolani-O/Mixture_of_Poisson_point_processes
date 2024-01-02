import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.figure import figaspect
import json
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
    parser.add_argument('--cuda', action='store_false', default=False, help='use CUDA (default: False)')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout applied to layers (default: 0.25)')
    parser.add_argument('--clip', type=float, default=0.2, help='gradient clip, -1 means no clip (default: 0.2)')
    parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')
    parser.add_argument('--levels', type=int, default=4, help='# of levels (default: 4)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=150, help='number of hidden units per layer (default: 150)')
    parser.add_argument('--batch_size', type=int_or_str, default='All', help='the batch size for training')
    parser.add_argument('--plot_lkhd', type=int, default=0, help='')
    parser.add_argument('--load', type=int, default=0, help='')
    parser.add_argument('--init_loss_load_epoch', type=int, default=1500, help='Which epoch to load for init loss')
    parser.add_argument('--init_map_load_epoch', type=int, default=1500, help='Which epoch to load for init map')
    parser.add_argument('--init_load_folder', type=str, default='', help='Which folder to load inits from')
    parser.add_argument('--init_load_subfolder_outputs', type=str, default='', help='Which subfolder to load outputs from')
    parser.add_argument('--init_load_subfolder_map', type=str, default='', help='Which subfolder to load map from')
    parser.add_argument('--init_load_subfolder_finetune', type=str, default='', help='Which subfolder to load finetune from')
    parser.add_argument('--train', type=int, default=0, help='')
    parser.add_argument('--reset_checkpoint', type=int, default=0, help='')
    parser.add_argument('--tau_psi', type=int, default=1, help='Value for tau_psi')
    parser.add_argument('--tau_beta', type=int, default=100, help='Value for tau_beta')
    parser.add_argument('--tau_s', type=int, default=10, help='Value for tau_s')
    parser.add_argument('--num_epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--notes', type=str, default='empty', help='Run notes')
    parser.add_argument('--K', type=int, default=50, help='Number of neurons')
    parser.add_argument('--R', type=int, default=5, help='Number of trials')
    parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
    parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
    parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
    parser.add_argument('--param_seed', type=int_or_str, default='', help='options are: seed (int), Truth (str)')
    parser.add_argument('--stage', type=str, default='finetune', help='options are: initialize_output, initialize_map, finetune, endtoend')
    return parser


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


def plot_spikes(binned, output_dir, x_offset=0):
    # Group entries by unique values of s[0]
    spikes = np.where(binned >= 1)
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset)/1000
        grouped_s.append((i, values))
    aspect_ratio = binned.shape[0] / binned.shape[1]
    w, h = figaspect(aspect_ratio)
    plt.figure(figsize=(w, h))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.savefig(os.path.join(output_dir, 'groundTruth_spikes.png'))


def plot_binned(binned, output_dir):
    # plot binned spikes
    _, ax = plt.subplots()
    ax.imshow(binned)
    ax.invert_yaxis()
    plt.savefig(os.path.join(output_dir, 'groundTruth_binned.png'))


def plot_intensity_and_latents(time, latent_factors, intensity, output_dir):
    # plot latent factors
    plt.figure()
    for i in range(latent_factors.shape[0]):
        plt.plot(time, latent_factors[i, :] + i)
    plt.savefig(os.path.join(output_dir, 'groundTruth_latent_factors.png'))

    # plot neuron intensities
    plt.figure()
    for i in range(intensity.shape[0]):
        plt.plot(time, intensity[i, :time.shape[0]] + i * 1)
    plt.savefig(os.path.join(output_dir, 'groundTruth_intensities.png'))


def plot_latent_coupling(latent_coupling, output_dir):
    # plot latent couplings
    plt.figure(figsize=(10, 10))
    sns.heatmap(latent_coupling, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of clusters')
    plt.savefig(os.path.join(output_dir, f'groundTruth_neuron_clusters.png'))


def plot_bsplines(B, time, output_dir):
    # plot bsplines
    start = 190
    num_basis = 10
    plt.figure()
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.savefig(os.path.join(output_dir, 'groundTruth_bsplines.png'))


def load_model_checkpoint(model_type, output_dir, folder_name, sub_folder_name, load_epoch):
    load_dir = os.path.join(output_dir, folder_name, sub_folder_name, 'models')
    file_name = os.path.join(load_dir, f'{model_type}_{load_epoch}.pth')
    if os.path.isfile(file_name):
        model = torch.load(file_name)
    else:
        print(f'No {model_type}_{load_epoch}.pth file found in {load_dir}')
        model = None
    return model


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


def create_relevant_files(output_dir, args, data_seed):
    with open(os.path.join(output_dir, 'log.txt'), 'w'):
        pass

    with open(os.path.join(output_dir, 'log_likelihoods_batch.json'), 'w+b') as file:
        file.write(b'[]')

    with open(os.path.join(output_dir, 'losses_batch.json'), 'w+b') as file:
        file.write(b'[]')

    with open(os.path.join(output_dir, 'log_likelihoods_train.json'), 'w+b') as file:
        file.write(b'[]')

    with open(os.path.join(output_dir, 'losses_train.json'), 'w+b') as file:
        file.write(b'[]')

    with open(os.path.join(output_dir, 'log_likelihoods_test.json'), 'w+b') as file:
        file.write(b'[]')

    with open(os.path.join(output_dir, 'losses_test.json'), 'w+b') as file:
        file.write(b'[]')

    command_str = (f"python src/psplines_gradient_method/main.py "
                   f"--K {args.K} --R {args.R} --L {args.L} --intensity_mltply {args.intensity_mltply} "
                   f"--intensity_bias {args.intensity_bias} --tau_beta {args.tau_beta} --tau_s {args.tau_s} "
                   f"--num_epochs {args.num_epochs} --notes {args.notes} "
                   f"--data_seed {data_seed} --param_seed {args.param_seed} --load_and_train 1")
    with open(os.path.join(output_dir, 'command.txt'), 'w') as file:
        file.write(command_str)


def write_log_and_model(output_str, output_dir, epoch, model=None, loss_function=None):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
        file.write(output_str)
    models_path = os.path.join(output_dir, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    if model is not None:
        torch.save(model, os.path.join(models_path, f'model_{epoch}.pth'))
    if loss_function is not None:
        torch.save(loss_function, os.path.join(models_path, f'loss_{epoch}.pth'))


def plot_outputs(latent_factors, cluster_attn, firing_attn, true_intensity, stim_time, output_dir, folder, epoch, batch=10):

    output_dir = os.path.join(output_dir, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height, width = [a/b*10 for a,b in zip(cluster_attn.shape, (50, 3))]
    plt.figure(figsize=(width,height))
    sns.heatmap(cluster_attn, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of cluster_attn')
    plt.savefig(os.path.join(output_dir, f'cluster_attn_{epoch}.png'))

    plt.figure(figsize=(5,height))
    sns.heatmap(firing_attn, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of firing_attn')
    plt.savefig(os.path.join(output_dir, f'firing_attn_{epoch}.png'))

    L, T = latent_factors.shape
    K = cluster_attn.shape[0]
    if K < batch:
        batch = cluster_attn.shape[0]

    weighted_firing = cluster_attn * firing_attn
    learned_intensity = weighted_firing @ latent_factors
    avg_lambda_intensities = np.mean(learned_intensity, axis=0)

    plt.figure()
    plt.plot(stim_time, avg_lambda_intensities)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, 'main_AvgLambdaIntensities.png'))

    plt.figure()
    for i in range(L):
        plt.plot(stim_time, latent_factors[i, :], label=f'Factor [{i}, :]')
        plt.title(f'Factors')
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, f'main_LatentFactors_{epoch}.png'))

    global_max = np.max(learned_intensity)
    upper_limit = global_max + batch * 0.01
    for i in range(0, K, batch):
        this_batch = batch if i + batch < K else K - i

        plt.figure(figsize=(10, 10))
        sorted_indices = sorted(range(this_batch), key=lambda j: np.argmax(cluster_attn[i + j]), reverse=True)
        for k, j in enumerate(sorted_indices):
            plt.subplot(2, 1, 1)
            plt.plot(stim_time, learned_intensity[i + j, :] + k * 0.01,
                     label=f'I={i + j}, C={np.argmax(cluster_attn[i + j])}, V={round(np.max(cluster_attn[i + j]), 2)}')
            plt.ylim(bottom=0, top=upper_limit)
            plt.subplot(2, 1, 2)
            plt.plot(stim_time, true_intensity[i + j, :stim_time.shape[0]] + k * 1, label=f'I={i + j}')
        plt.subplot(2, 1, 1)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'main_LambdaIntensities_Trial_batch{i}.png'))


def write_losses(list, name, metric, output_dir, starts_out_empty):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    else:
        file_name = 'losses'
    file_name = f'{file_name}_{name.lower()}.json'
    with open(os.path.join(output_dir, file_name), 'r+b') as file:
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


def plot_losses(true_likelihood, output_dir, name, metric, cutoff):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    else:
        file_name = 'losses'
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
    plt.savefig(os.path.join(plt_path, f'{metric}_{name}_Trajectories.png'))


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value