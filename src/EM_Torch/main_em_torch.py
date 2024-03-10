import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.LikelihoodModel import LikelihoodModel
from src.EM_Torch.general_functions import load_model_checkpoint, softplus, plot_spikes, \
    plot_intensity_and_latents, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_losses, CustomDataset
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

args = get_parser().parse_args()
if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

# args.n_trials = 5  # R
# args.n_configs = 5  # C
# args.n_trial_samples = 4  # M
# args.n_config_samples = 4  # N
# args.K = 3  # K

# args.folder_name = ''
# args.load = True
# args.load_epoch = 0
# args.data_seed = 0

# args.notes = 'Full'
# args.lr = 0.001
# args.batch_size = 'All'
# args.param_seed = 'InitAllGroundTruth'

print('Start')
outputs_folder = 'outputs'
# outputs_folder = '../../outputs'
output_dir = os.path.join(os.getcwd(), outputs_folder)
# Set the random seed manually for reproducibility.
np.random.seed(args.data_seed)
# if args.param_seed != 'TRUTH':
#     torch.manual_seed(args.param_seed)
if args.load:
    output_dir = os.path.join(output_dir, args.folder_name)
    model, data = load_model_checkpoint(output_dir, args.load_epoch)
else:
    # Ground truth data
    data = DataAnalyzer().initialize(A=args.A, intensity_mltply=args.intensity_mltply, intensity_bias=args.intensity_bias)

    num_factors = data.beta.shape[0]
    model = LikelihoodELBOModel(data.time, args.n_trial_samples, args.n_config_samples)
    model.init_ground_truth(num_factors,
                            torch.tensor(data.beta), torch.tensor(data.alpha),
                            torch.tensor(data.theta), torch.tensor(data.pi),
                            torch.tensor(data.config_peak_offset_stdevs),
                            torch.tensor(data.trial_peak_offset_covar_ltri))
    # model.init_ground_truth(num_factors, torch.zeros_like(torch.tensor(data.beta)).float())
    # model.init_ground_truth(num_factors, torch.tensor(data.beta).float())
    # la = int(num_factors/args.A)
    # factor_indcs = [i*la for i in range(args.A)]
    # model.init_from_data(Y=torch.tensor(Y_train).float(), neuron_factor_access=torch.tensor(factor_access_train).float(),
    #                      factor_indcs=factor_indcs)
    # model.init_random(num_factors)

    args.folder_name = (
        f'{args.param_seed}_dataSeed{args.data_seed}_K{args.K}_R{args.n_trials}_A{args.A}_C{args.n_configs}'
        f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauSigma1{args.tau_sigma1}_tauSigma2{args.tau_sigma2}'
        f'_iters{args.num_epochs}_BatchSize{args.batch_size}_lr{args.lr}_notes-{args.notes}')
    output_dir = os.path.join(output_dir, args.folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Training data
Y_train, stim_time, factor_access_train = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)
(intensities_train, factor_assignment_train, factor_assignment_onehot_train, config_offsets_train,
 trial_offsets_train) = data.get_data_ground_truth()
# Validation data
Y_test, _, factor_access_test = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)
(intensities_test, factor_assignment_test, factor_assignment_onehot_test, config_offsets_test,
 trial_offsets_test) = data.get_data_ground_truth()

# true_likelihood_train = data.compute_log_likelihood(Y_train, intensities_train, factor_assignment_train,
#                                                     config_offsets_train, trial_offsets_train)
# true_likelihood_test = data.compute_log_likelihood(Y_test, intensities_test, factor_assignment_test,
#                                                    config_offsets_test, trial_offsets_test)

# initialize the model with ground truth params
true_model = LikelihoodModel(stim_time)
true_model.init_params(torch.tensor(data.beta), torch.tensor(data.alpha),
                       torch.tensor(data.theta), torch.tensor(data.pi),
                       torch.tensor(data.config_peak_offset_stdevs),
                       torch.tensor(data.trial_peak_offset_covar_ltri),
                       args.n_configs, args.K)

true_model.eval()
with torch.no_grad():
    true_model.init_ground_truth(torch.tensor(config_offsets_train), torch.tensor(trial_offsets_train))
    likelihood_term_train, entropy_term_train = true_model.forward(torch.tensor(Y_train), torch.tensor(factor_access_train), args.A)
    true_ELBO_train = (1/(Y_train.shape[0] * Y_train.shape[-1])) * likelihood_term_train + (1/Y_train.shape[-1]) * entropy_term_train

    true_model.init_ground_truth(torch.tensor(config_offsets_test), torch.tensor(trial_offsets_test))
    likelihood_term_test, entropy_term_test = true_model.forward(torch.tensor(Y_test), torch.tensor(factor_access_test), args.A)
    true_ELBO_test = (1/(Y_test.shape[0] * Y_test.shape[-1])) * likelihood_term_test + (1/Y_test.shape[-1]) * entropy_term_test

output_str = (
    f"True ELBO Training: {true_ELBO_train},\n"
    f"True ELBO Test: {true_ELBO_test}\n\n")
create_relevant_files(output_dir, args, output_str, data)
plot_spikes(Y_train, output_dir, model.dt.item(), 'train')
plot_spikes(Y_test, output_dir, model.dt.item(), 'test')
plot_intensity_and_latents(data.time, softplus(data.beta), data.neuron_intensities, output_dir)

# Instantiate the dataset and dataloader
dataset = CustomDataset(torch.tensor(Y_train), torch.tensor(factor_access_train))
if args.batch_size == 'All':
    args.batch_size = Y_train.shape[0]
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.lr)
print(f'folder_name: {args.folder_name}')
print(output_str)

if __name__ == "__main__":
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
    start_time = time.time()
    for epoch in range(args.num_epochs):

        model.train()
        optimizer.zero_grad()
        warped_factors = model.warp_all_latent_factors_for_all_trials(args.n_configs, args.n_trials)
        penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_budget, args.tau_sigma1, args.tau_sigma2)
        entropy_term = 0
        likelihood_term = 0
        for Y, access in dataloader:
            likelihood_term += (1/(Y.shape[0]*Y.shape[-1])) * model.compute_log_elbo(Y, access, warped_factors, args.A)
            batch_weights = Y_train.shape[0]//args.batch_size
            batch_add = 1 if Y_train.shape[0] % args.batch_size > 0 else 0
            entropy_term += (1/((batch_weights+batch_add)*Y.shape[-1])) * model.compute_offset_entropy_terms()
            # because entropy term depends on W_C_tensor, which depends on Y_kc
        loss = -(likelihood_term + entropy_term + penalty_term)
        loss.backward()
        optimizer.step()
        losses_train.append((likelihood_term + entropy_term + penalty_term).item())
        log_likelihoods_train.append((likelihood_term + entropy_term).item())

        if epoch % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                beta_mses.append(F.mse_loss(model.beta, torch.tensor(data.beta)).item())
                alpha_mses.append(F.mse_loss(model.alpha, torch.tensor(data.alpha)).item())
                theta_mses.append(F.mse_loss(model.theta, torch.tensor(data.theta)).item())
                pi_mses.append(F.mse_loss(model.pi, torch.tensor(data.pi)).item())
                stdevs_mses.append(F.mse_loss(model.config_peak_offset_stdevs, torch.tensor(data.config_peak_offset_stdevs)).item())
                ltri_mses.append(F.mse_loss(model.trial_peak_offset_covar_ltri, torch.tensor(data.trial_peak_offset_covar_ltri)).item())

                likelihood_term = (1/(Y_test.shape[0]*Y_test.shape[-1])) * model.compute_log_elbo(torch.tensor(Y_test), torch.tensor(factor_access_test), warped_factors, args.A)
                entropy_term = (1/Y_test.shape[-1]) * model.compute_offset_entropy_terms()
                losses_test.append((likelihood_term + entropy_term + penalty_term).item())
                log_likelihoods_test.append((likelihood_term + entropy_term).item())

        if epoch % args.log_interval == 0 or epoch == args.num_epochs - 1:
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
                warped_factors = None  # model.warp_all_latent_factors_for_all_trials(args.n_configs, args.n_trials).numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"Loss test: {cur_loss_test:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"lr: {args.lr:.5f}, smoothness_budget: {smoothness_budget_constrained.T}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model)
            plot_outputs(latent_factors, warped_factors, stim_time, output_dir, 'Train', epoch)
            is_empty = epoch == 0
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
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood')
            plot_losses(None, output_dir, 'Train', 'Loss', 20)
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood')
            plot_losses(None, output_dir, 'Test', 'Loss', 1)
            plot_losses(None, output_dir, 'Test', 'beta_MSE')
            plot_losses(None, output_dir, 'Test', 'alpha_MSE')
            plot_losses(None, output_dir, 'Test', 'theta_MSE')
            plot_losses(None, output_dir, 'Test', 'pi_MSE')
            plot_losses(None, output_dir, 'Test', 'stdevs_MSE')
            plot_losses(None, output_dir, 'Test', 'ltri_MSE')
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
            start_time = time.time()
