import os
import sys
sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import load_model_checkpoint, softplus, plot_spikes, \
    plot_intensity_and_latents, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_losses
import numpy as np
import time
import torch
import torch.nn.functional as F


args = get_parser().parse_args()
if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

# args.n_trials = 5  # R
# args.n_configs = 5  # C
# args.n_trial_samples = 4  # M
# args.n_config_samples = 4  # N
# args.K = 10  # K

args.folder_name = 'paramSeedTRUTH_dataSeed1597527101_L3_K30_R10_A3_C15_int.mltply25_int.add1_tauBeta0_tauSigma10_tauSigma20_iters5000_notes-Full_InitBeta&PiFromData_lr0.01'
args.load = True
args.load_epoch = 250
args.data_seed = 1597527101

# args.param_seed = 'TRUTH'
# args.notes = 'Full_InitBeta&PiFromData_lr0.01'
args.lr = 0.01
# args.tau_beta = 0
# args.tau_sigma1 = 0
# args.tau_sigma2 = 0
# args.tau_budget = 0

print('Start')
outputs_folder = 'outputs'
# outputs_folder = '../../outputs'
output_dir = os.path.join(os.getcwd(), outputs_folder)
# Set the random seed manually for reproducibility.
np.random.seed(args.data_seed)
if args.load:
    output_dir = os.path.join(output_dir, args.folder_name)
    model, output_str, data = load_model_checkpoint(output_dir, args.load_epoch)
    # Training data
    Y_train, stim_time, factor_access_train, intensities_train = data.sample_data(K=args.K, A=args.A,
                                                                                  n_configs=args.n_configs,
                                                                                  n_trials=args.n_trials)
    # Validation data
    Y_test, _, factor_access_test, intensities_test = data.sample_data(K=10, A=args.A, n_configs=5, n_trials=5)
    output_str_split = output_str.split(':')
    true_ELBO_train = float(output_str_split[3].split(',')[0].strip())
    true_ELBO_test = float(output_str_split[4].split('\n')[0].strip())
    start_epoch = args.load_epoch+1
else:
    # Ground truth data
    data = DataAnalyzer().initialize(A=args.A, intensity_mltply=args.intensity_mltply,
                                     intensity_bias=args.intensity_bias)
    # Training data
    Y_train, stim_time, factor_access_train, intensities_train = data.sample_data(K=args.K, A=args.A,
                                                                                  n_configs=args.n_configs,
                                                                                  n_trials=args.n_trials)
    # Validation data
    Y_test, _, factor_access_test, intensities_test = data.sample_data(K=10, A=args.A, n_configs=5, n_trials=5)

    true_likelihood_train = data.compute_log_likelihood(Y_train, intensities_train)
    true_likelihood_test = data.compute_log_likelihood(Y_test, intensities_test)

    # initialize the model with training data and ground truth params
    model = LikelihoodELBOModel(stim_time, args.n_trial_samples, args.n_config_samples)
    model.init_ground_truth(data.beta.shape[0],
                            torch.tensor(data.beta).float(), torch.tensor(data.alpha).float(),
                            torch.tensor(data.theta).float(), torch.tensor(data.pi).float(),
                            torch.tensor(data.config_peak_offset_stdevs).float(),
                            torch.tensor(data.trial_peak_offset_covar_ltri).float())

    model.eval()
    with torch.no_grad():
        true_ELBO_train = model.compute_log_elbo(torch.tensor(Y_train), torch.tensor(factor_access_train),
                                                 torch.arange(args.n_configs)) + model.compute_offset_entropy_terms()
        true_ELBO_test = model.compute_log_elbo(torch.tensor(Y_test), torch.tensor(factor_access_test),
                                                torch.arange(args.n_configs)) + model.compute_offset_entropy_terms()

    start_epoch = 0

    # # Remove this lines
    # model.init_ground_truth(data.beta.shape[0], torch.zeros_like(data.beta).float())
    # model.init_ground_truth(data.beta.shape[0], torch.tensor(data.beta).float())
    model.init_from_data(Y=torch.tensor(Y_train).float(),
                         neuron_factor_access=torch.tensor(factor_access_train).float())

    if args.param_seed != 'TRUTH':
        torch.manual_seed(args.param_seed)
        model.init_random(factor_access_train.shape[2])
    args.folder_name = (f'paramSeed{args.param_seed}_dataSeed{args.data_seed}_L{args.L}_K{args.K}_R{args.n_trials}_A{args.A}_C{args.n_configs}'
                        f'_int.mltply{args.intensity_mltply}_int.add{args.intensity_bias}'
                        f'_tauBeta{args.tau_beta}_tauSigma1{args.tau_sigma1}_tauSigma2{args.tau_sigma2}_iters{args.num_epochs}'
                        f'_notes-{args.notes}')
    output_dir = os.path.join(output_dir, args.folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_str = (
        f"True likelihood Training: {true_likelihood_train},\n"
        f"True likelihood Test: {true_likelihood_test},\n"
        f"True ELBO Training: {true_ELBO_train},\n"
        f"True ELBO Test: {true_ELBO_test}\n\n")
    create_relevant_files(output_dir, args, output_str, data)
    plot_spikes(Y_train, output_dir, model.dt.item(), 'train')
    plot_spikes(Y_test, output_dir, model.dt.item(), 'test')
    plot_intensity_and_latents(data.time, softplus(data.beta), data.neuron_intensities, output_dir)

optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.lr)
print(f'folder_name: {args.folder_name}')
print(output_str)

if __name__ == "__main__":
    # log_likelihoods = []
    # losses = []
    log_likelihoods_train = []
    losses_train = []
    log_likelihoods_test = []
    losses_test = []
    beta_mses = []
    alpha_mses = []
    theta_mses = []
    pi_mses = []
    stdevs_mses = []
    ltri_mses = []
    total_time = 0
    # For debugging
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        start_time = time.time()  # Record the start time of the epoch

        model.train()
        optimizer.zero_grad()
        likelihood_term, entropy_term, penalty_term = model.forward(Y=torch.tensor(Y_train),
                                                                    neuron_factor_access=torch.tensor(factor_access_train),
                                                                    config_indcs=torch.arange(args.n_configs),
                                                                    tau_beta=args.tau_beta, tau_budget=args.tau_budget,
                                                                    tau_sigma1=args.tau_sigma1, tau_sigma2=args.tau_sigma2)
        loss = -(likelihood_term + entropy_term + penalty_term)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            losses_train.append((likelihood_term + entropy_term + penalty_term).item())
            log_likelihoods_train.append((likelihood_term + entropy_term).item())
            beta_mses.append(F.mse_loss(model.beta, torch.tensor(data.beta)).item())
            alpha_mses.append(F.mse_loss(model.alpha, torch.tensor(data.alpha)).item())
            theta_mses.append(F.mse_loss(model.theta, torch.tensor(data.theta)).item())
            pi_mses.append(F.mse_loss(model.pi, torch.tensor(data.pi)).item())
            stdevs_mses.append(F.mse_loss(model.config_peak_offset_stdevs, torch.tensor(data.config_peak_offset_stdevs)).item())
            ltri_mses.append(F.mse_loss(model.trial_peak_offset_covar_ltri, torch.tensor(data.trial_peak_offset_covar_ltri)).item())

            likelihood_term, entropy_term, penalty_term = model.forward(Y=torch.tensor(Y_test),
                                                                        neuron_factor_access=torch.tensor(factor_access_test),
                                                                        config_indcs=torch.arange(Y_test.shape[-1]),
                                                                        tau_beta=args.tau_beta, tau_budget=args.tau_budget,
                                                                        tau_sigma1=args.tau_sigma1, tau_sigma2=args.tau_sigma2)
            losses_test.append((likelihood_term + entropy_term + penalty_term).item())
            log_likelihoods_test.append((likelihood_term + entropy_term).item())

        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            cur_loss_test = losses_test[-1]
            with torch.no_grad():
                smoothness_budget_constrained = F.softmax(model.smoothness_budget, dim=0).numpy()
                latent_factors = F.softplus(model.beta).numpy()
                warped_factors = None # model.warp_all_latent_factors_for_all_trials(args.n_configs, args.n_trials).numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"Loss test: {cur_loss_test:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"lr: {args.lr:.5f}, smoothness_budget: {smoothness_budget_constrained.T}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model)
            plot_outputs(latent_factors, warped_factors, stim_time, output_dir, 'Train', epoch)
            is_empty = start_epoch == 0 and epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            write_losses(beta_mses, 'Test', 'beta_MSE', output_dir, is_empty)
            write_losses(alpha_mses, 'Test', 'alpha_MSE', output_dir, is_empty)
            write_losses(theta_mses, 'Test', 'theta_MSE', output_dir, is_empty)
            write_losses(pi_mses, 'Test', 'pi_MSE', output_dir, is_empty)
            write_losses(stdevs_mses, 'Test', 'stdevs_MSE', output_dir, is_empty)
            write_losses(ltri_mses, 'Test', 'ltri_MSE', output_dir, is_empty)
            # write_losses(log_likelihoods, 'Batch', 'Likelihood', output_dir, is_empty)
            # write_losses(losses, 'Batch', 'Loss', output_dir, is_empty)
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Train', 'Loss', 20)
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Test', 'Loss', 20)
            plot_losses(None, output_dir, 'Test', 'beta_MSE', 20)
            plot_losses(None, output_dir, 'Test', 'alpha_MSE', 20)
            plot_losses(None, output_dir, 'Test', 'theta_MSE', 20)
            plot_losses(None, output_dir, 'Test', 'pi_MSE', 20)
            plot_losses(None, output_dir, 'Test', 'stdevs_MSE', 20)
            plot_losses(None, output_dir, 'Test', 'ltri_MSE', 20)
            # plot_losses(true_ELBO_train/args.n_configs, output_dir, 'Batch', 'Likelihood', 100)
            # plot_losses(0, output_dir, 'Batch', 'Loss', 100)
            log_likelihoods = []
            losses = []
            log_likelihoods_train = []
            losses_train = []
            log_likelihoods_test = []
            losses_test = []
            beta_mses = []
            alpha_mses = []
            theta_mses = []
            pi_mses = []
            stdevs_mses = []
            ltri_mses = []
            print(output_str)
