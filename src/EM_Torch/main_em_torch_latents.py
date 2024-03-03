import os
import sys
sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.LikelihoodModel import LikelihoodModel
from src.EM_Torch.general_functions import load_model_checkpoint, softplus, plot_spikes, \
    plot_intensity_and_latents, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_losses
import numpy as np
import time
import torch
import torch.nn.functional as F


args = get_parser().parse_args()

args.n_trials = 2  # R
args.n_configs = 3  # C

# args.folder_name = ''
# args.load = True
# args.load_epoch = 1499

args.notes = 'Full_Ground-Truth-Init'

if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

print('Start')
# Set the random seed manually for reproducibility.
np.random.seed(args.data_seed)
# Ground truth data
data = DataAnalyzer().initialize(A=args.A, intensity_mltply=args.intensity_mltply, intensity_bias=args.intensity_bias)
# Training data
Y_train, stim_time, factor_access_train, intensities_train = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)
# Validation data
Y_test, _, factor_access_test, intensities_test = data.sample_data(K=args.K, A=args.A, n_configs=args.n_configs, n_trials=args.n_trials)

# initialize the model with training data and ground truth params
model = LikelihoodModel(stim_time)
model.init_ground_truth(torch.tensor(data.beta).float(), torch.tensor(data.alpha).float(),
                        torch.tensor(data.theta).float(), torch.tensor(data.pi).float(),
                        torch.tensor(data.config_peak_offset_stdevs).float(),
                        torch.tensor(data.trial_peak_offset_covar_ltri).float(),
                        args.n_configs, args.n_trials)

model.eval()
with torch.no_grad():
    true_ELBO_train = model.compute_log_elbo(torch.tensor(Y_train), torch.tensor(factor_access_train), torch.arange(args.n_configs)) + model.compute_offset_entropy_terms()
    true_ELBO_test = model.compute_log_elbo(torch.tensor(Y_test), torch.tensor(factor_access_test), torch.arange(args.n_configs)) + model.compute_offset_entropy_terms()

output_dir = os.path.join(os.getcwd(), 'outputs')
output_str = ''
start_epoch = 0

if args.load:
    output_dir = os.path.join(output_dir, args.folder_name)
    model, output_str, data = load_model_checkpoint(output_dir, args.load_epoch)
    start_epoch = args.load_epoch
else:
    args.folder_name = (f'paramSeed{args.param_seed}_dataSeed{args.data_seed}_L{args.L}_K{args.K}_R{args.n_trials}_A{args.A}_C{args.n_configs}'
                        f'_int.mltply{args.intensity_mltply}_int.add{args.intensity_bias}_iters{args.num_epochs}_notes-{args.notes}')
    output_dir = os.path.join(output_dir, args.folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_str = (
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
    log_likelihoods_train = []
    log_likelihoods_test = []
    total_time = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        start_time = time.time()  # Record the start time of the epoch

        model.train()
        optimizer.zero_grad()
        likelihood_term, entropy_term = model.forward(Y=torch.tensor(Y_train),
                                                      neuron_factor_access=torch.tensor(factor_access_train),
                                                      config_indcs=torch.arange(args.n_configs))
        loss = -(likelihood_term + entropy_term)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            log_likelihoods_train.append((likelihood_term + entropy_term).item())

            likelihood_term, entropy_term = model.forward(Y=torch.tensor(Y_test),
                                                          neuron_factor_access=torch.tensor(factor_access_test),
                                                          config_indcs=torch.arange(args.n_configs))
            log_likelihoods_test.append((likelihood_term + entropy_term).item())

        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            with torch.no_grad():
                smoothness_budget_constrained = F.softmax(model.smoothness_budget, dim=0).numpy()
                latent_factors = F.softplus(model.beta).numpy()
                warped_factors = None # model.warp_all_latent_factors_for_all_trials(args.n_configs, args.n_trials).numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Log Likelihood train: {cur_log_likelihood_train:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"lr: {args.lr:.5f}, smoothness_budget: {smoothness_budget_constrained.T}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model)
            plot_outputs(latent_factors, warped_factors, stim_time, output_dir, 'Train', epoch)
            is_empty = start_epoch == 0 and epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            # write_losses(log_likelihoods, 'Batch', 'Likelihood', output_dir, is_empty)
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood', 20)
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood', 20)
            # plot_losses(true_ELBO_train/args.n_configs, output_dir, 'Batch', 'Likelihood', 100)
            log_likelihoods = []
            log_likelihoods_train = []
            log_likelihoods_test = []
            print(output_str)
