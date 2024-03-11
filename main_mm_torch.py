import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
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
# args.data_seed = 3259374874

# args.notes = 'Full'
# args.lr = 0.001
# args.batch_size = 'All'
# args.param_seed = 'InitBetaGroundTruth'

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
    model = LikelihoodModel(data.time)
    model.init_params(torch.tensor(data.beta), torch.tensor(data.alpha),
                       torch.tensor(data.theta), torch.tensor(data.pi),
                       torch.tensor(data.config_peak_offset_stdevs),
                       torch.tensor(data.trial_peak_offset_covar_ltri),
                       args.n_configs, args.K)
    # model.init_random
    args.folder_name = (
        f'{args.param_seed}_dataSeed{args.data_seed}_K{args.K}_R{args.n_trials}_A{args.A}_C{args.n_configs}'
        f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauSigma1{args.tau_sigma1}_tauSigma2{args.tau_sigma2}'
        f'_iters{args.num_epochs}_BatchSize{args.batch_size}_lr{args.lr}_notes-{args.notes}')
    output_dir = os.path.join(output_dir, args.folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Training data
Y_train, stim_time, factor_access_train = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)
(intensities_train, factor_assignment_train, factor_assignment_onehot_train, neuron_gains_train, config_offsets_train,
 trial_offsets_train) = data.get_data_ground_truth()
# Validation data
Y_test, _, factor_access_test = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)
(intensities_test, factor_assignment_test, factor_assignment_onehot_test, neuron_gains_test, config_offsets_test,
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
    likelihood_term_train, entropy_term_train, model_factor_assignment_train = true_model.forward(torch.tensor(Y_train), torch.tensor(factor_access_train), args.A)
    true_ELBO_train = (1/(Y_train.shape[0] * Y_train.shape[-1])) * likelihood_term_train + (1/Y_train.shape[-1]) * entropy_term_train

    true_model.init_ground_truth(torch.tensor(config_offsets_test), torch.tensor(trial_offsets_test))
    likelihood_term_test, entropy_term_test, model_factor_assignment_test = true_model.forward(torch.tensor(Y_test), torch.tensor(factor_access_test), args.A)
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
    log_likelihoods_test = []
    total_time = 0
    # For debugging
    # torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    for epoch in range(args.num_epochs):

        model.train()
        optimizer.zero_grad()
        warped_factors = model.warp_all_latent_factors_for_all_trials()
        entropy_term = (1/Y_train.shape[-1]) * model.compute_offset_entropy_terms()
        # There is no W_C_tensor (which depends on Y_kc) here
        likelihood_term = 0
        for Y, access in dataloader:
            likelihood_term += (1/(Y.shape[0]*Y.shape[-1])) * model.compute_log_elbo(Y, access, warped_factors, args.A)
        loss = -(likelihood_term + entropy_term)
        loss.backward()
        optimizer.step()
        log_likelihoods_train.append((likelihood_term + entropy_term).item())

        if epoch % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                likelihood_term = (1/(Y_test.shape[0]*Y_test.shape[-1])) * model.compute_log_elbo(torch.tensor(Y_test), torch.tensor(factor_access_test), warped_factors, args.A)
                log_likelihoods_test.append((likelihood_term + entropy_term).item())

        if epoch % args.log_interval == 0 or epoch == args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            with torch.no_grad():
                latent_factors = F.softplus(model.beta).numpy()
                warped_factors = None # model.warp_all_latent_factors_for_all_trials(args.n_configs, args.n_trials).numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Log Likelihood train: {cur_log_likelihood_train:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"lr: {args.lr:.5f}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model)
            is_empty = epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood')
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood')
            log_likelihoods_train = []
            log_likelihoods_test = []
            print(output_str)
            start_time = time.time()
