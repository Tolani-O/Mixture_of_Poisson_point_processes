import os
import sys
sys.path.append(os.path.abspath('../EM_Torch'))
import pickle
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.SpikeTrainModel_EM import LikelihoodModel
from src.psplines_gradient_method.general_functions import plot_outputs, plot_spikes, plot_intensity_and_latents, \
    plot_likelihoods, int_or_str, write_outputs
import numpy as np
import time
import argparse
import ijson
import torch
import torch.nn.functional as F


def main(K, R, L, intensity_mltply, intensity_bias, tau_psi, tau_beta, tau_s, beta_first, notes, num_epochs,
         data_seed, param_seed, load_only, load_and_train):

    folder_name = (f'paramSeed{param_seed}_dataSeed{data_seed}_L{L}_K{K}_R{R}'
                   f'_int.mltply{intensity_mltply}_int.add{intensity_bias}'
                   f'_tauBeta{tau_beta}_tauS{tau_s}_iters{num_epochs}_betaFirst{beta_first}'
                   f'_notes-{notes}')
    print(f'folder_name: {folder_name}')
    output_dir = os.path.join(os.getcwd(), 'outputs', folder_name)

    # np.random.seed(data_seed)
    n_trials = 15
    n_configs = 10
    A = 2
    data = DataAnalyzer().initialize(K=K, A=A, intensity_mltply=intensity_mltply, intensity_bias=intensity_bias, n_trials=n_trials, n_configs=n_configs)
    Y, stim_time, factor_access = data.sample_data()
    n_trial_samples = 10 # m
    n_config_samples = 10 # n
    model = LikelihoodModel().initialize(Y, stim_time, factor_access, n_trial_samples, n_config_samples)
    factors = np.exp() @ data.B
    model.init_ground_truth(torch.tensor(data.beta), torch.tensor(data.alpha), torch.tensor(data.theta), torch.tensor(data.pi),
                            torch.tensor(data.config_peak_offset_stdevs), torch.tensor(data.trial_peak_offset_covar_ltri))
    optimizer_nn = torch.optim.Adam(model.parameters(), lr=0.01)

    true_likelihood = data.compute_log_likelihood()
    true_ELBO = model.compute_log_elbo() + model.compute_offset_entropy_terms()
    print(f"True likelihood: {true_likelihood}")
    start_epoch = 0

    # if load_only or load_and_train:
    #     with open(os.path.join(output_dir, 'model.pkl'), 'rb') as model_file:
    #         model = pickle.load(model_file)
    #     with open(os.path.join(output_dir, 'log_likelihoods.json'), 'rb') as file:
    #         for item in ijson.items(file, 'item'):
    #             start_epoch += 1
    #
    # if load_only:
    #     metrics_results = {}
    #     training_results = {
    #         "model": model,
    #         "data": data
    #     }
    #     return training_results, metrics_results
    #
    # if not load_and_train:
    #
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     with open(os.path.join(output_dir, 'log.txt'), 'w'):
    #         pass
    #
    #     with open(os.path.join(output_dir, 'log_likelihoods.json'), 'w+b') as file:
    #         file.write(b'[]')
    #
    #     command_str = (f"python src/psplines_gradient_method/main.py "
    #                    f"--K {K} --R {R} --L {L} --intensity_mltply {intensity_mltply} --intensity_bias {intensity_bias} "
    #                    f"--tau_beta {tau_beta} --tau_s {tau_s} --num_epochs {num_epochs} --beta_first {beta_first} --notes {notes} "
    #                    f"--data_seed {data_seed} --param_seed {param_seed} --load_and_train 1")
    #     with open(os.path.join(output_dir, 'command.txt'), 'w') as file:
    #         file.write(command_str)
    #
    #     binned, stim_time = data.sample_data()
    #     plot_spikes(binned, output_dir)
    #     plot_intensity_and_latents(data.time, data.latent_factors, data.intensity, output_dir)
    #     Y = binned  # K x T
    #     degree = 3
    #     if param_seed == 'TRUTH':
    #         model = LikelihoodModel(Y, stim_time).initialize_for_time_warping(L, degree)
    #         model.init_ground_truth(data.latent_factors, data.neuron_coupling)
    #     else:
    #         np.random.seed(param_seed)
    #         model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

    losses = []
    log_likelihoods = []
    alpha_loss_increase = []
    gamma_loss_increase = []
    c_loss_increase = []
    d2_loss_increase = []
    zeta_loss_increase = []
    chi_loss_increase = []
    alpha_learning_rate = []
    gamma_learning_rate = []
    c_learning_rate = []
    d2_learning_rate = []
    zeta_learning_rate = []
    chi_learning_rate = []
    alpha_iters = []
    gamma_iters = []
    c_iters = []
    d2_iters = []
    zeta_iters = []
    chi_iters = []
    total_time = 0
    epoch_time = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()  # Record the start time of the epoch

        optimizer_nn.zero_grad()
        loss = model.forward(10, 10 ,10)
        loss.backward()
        optimizer_nn.step()

        # loss = result["loss"]
        # log_likelihood = result["log_likelihood"]
        losses.append(loss)
        # log_likelihoods.append(log_likelihood)

        end_time = time.time()  # Record the end time of the epoch
        elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
        epoch_time += elapsed_time  # Record the elapsed time for the epoch
        total_time += elapsed_time  # Calculate the total time for training

        if epoch % 10 == 0:
            s2_norm = F.softmax(model.smoothness_budget, dim=0)
            output_str = (f"Epoch: {epoch}, Loss: {-loss.detach()}, "
                          f"Epoch Time: {epoch_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs\n"
                          f"s_norm: {s2_norm.detach().t()}\n")
            # output_str = (f"Epoch: {epoch}, Loss: {loss}, Log Likelihood: {log_likelihood}, "
            #               f"Epoch Time: {epoch_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs\n"
            #               f"s_norm: {s2_norm.T}\n")
            print(output_str)
            epoch_time = 0
            # plot_outputs(model, data, output_dir, epoch)
            # write_outputs(output_dir, log_likelihoods, epoch==0, model, output_str)
            # log_likelihoods = []

    # plot_outputs(model, data, output_dir, epoch)
    # write_outputs(output_dir, log_likelihoods, 0, model, output_str)

    metrics_results = {
        "losses": losses,
        "log_likelihoods": log_likelihoods,
        "alpha_loss_increase": alpha_loss_increase,
        "gamma_loss_increase": gamma_loss_increase,
        "c_loss_increase": c_loss_increase,
        "d2_loss_increase": d2_loss_increase,
        "zeta_loss_increase": zeta_loss_increase,
        "chi_loss_increase": chi_loss_increase,
        "alpha_learning_rate": alpha_learning_rate,
        "gamma_learning_rate": gamma_learning_rate,
        "c_learning_rate": c_learning_rate,
        "d2_learning_rate": d2_learning_rate,
        "zeta_learning_rate": zeta_learning_rate,
        "chi_learning_rate": chi_learning_rate,
        "alpha_iters": alpha_iters,
        "gamma_iters": gamma_iters,
        "c_iters": c_iters,
        "d2_iters": d2_iters,
        "zeta_iters": zeta_iters,
        "chi_iters": chi_iters
    }

    training_results = {
        "model": model,
        "data": data
    }
    return training_results, metrics_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Python script from the command line.')

    parser.add_argument('--plot_lkhd', type=int, default=0, help='')
    parser.add_argument('--load_only', type=int, default=0, help='')
    parser.add_argument('--load_and_train', type=int, default=0, help='')
    parser.add_argument('--tau_psi', type=int, default=1, help='Value for tau_psi')
    parser.add_argument('--tau_beta', type=int, default=5000, help='Value for tau_beta')
    parser.add_argument('--tau_s', type=int, default=15000, help='Value for tau_s')
    parser.add_argument('--num_epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--beta_first', type=int, default=0, help='Whether to update beta first or G first')
    parser.add_argument('--notes', type=str, default='empty', help='Run notes')
    parser.add_argument('--K', type=int, default=100, help='Number of neurons')
    parser.add_argument('--R', type=int, default=50, help='Number of trials')
    parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
    parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
    parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
    parser.add_argument('--param_seed', type=int_or_str, default='', help='')

    args = parser.parse_args()
    plot_lkhd = args.plot_lkhd
    K = args.K
    R = args.R
    L = args.L
    intensity_mltply = args.intensity_mltply
    intensity_bias = args.intensity_bias
    tau_psi = args.tau_psi
    tau_beta = args.tau_beta
    tau_s = args.tau_s
    num_epochs = args.num_epochs
    beta_first = args.beta_first
    notes = args.notes
    load_only = args.load_only
    load_and_train = args.load_and_train
    param_seed = args.param_seed
    if param_seed == '':
        param_seed = np.random.randint(0, 2 ** 32 - 1)
    data_seed = np.random.randint(0, 2 ** 32 - 1)

    # override
    data_seed = 4181387928
    # param_seed = 'TRUTH'
    # load_only = 1
    # plot_lkhd = 1
    # load_and_train = 1

    if plot_lkhd:
        np.random.seed(data_seed)
        true_data = DataAnalyzer().initialize(K=K, R=R, intensity_mltply=intensity_mltply, intensity_bias=intensity_bias, max_offset=0)
        plot_likelihoods(true_data, K, R, L, intensity_mltply, intensity_bias, data_seed)
        sys.exit()
    training_results, metrics_results = main(K, R, L, intensity_mltply, intensity_bias, tau_psi, tau_beta, tau_s,
                                             beta_first, notes, num_epochs, data_seed, param_seed,
                                             load_only, load_and_train)
