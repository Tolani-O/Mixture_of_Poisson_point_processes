import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import load_model_checkpoint, plot_factor_assignments, plot_spikes, \
    plot_intensity_and_latents, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_losses, CustomDataset, load_tensors
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

args = get_parser().parse_args()
if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
args.data_seed = np.random.randint(0, 2 ** 32 - 1)
outputs_folder = 'outputs'

# args.n_trials = 5  # R
# args.n_configs = 1  # C
# args.n_trial_samples = 2  # N
# args.K = 3  # K
# args.A = 1  # A
# args.tau_beta = 0
# args.tau_beta = 0
# args.tau_config = 0
# args.tau_sigma = 0


args.folder_name = 'TrueInit+MinorPenalty1+Full_dataSeed4198318570_K30_R10_A2_C2_R10_tauBeta1_tauConfig1_tauSigma1_iters50000_BatchSizeAll_lr0.001_patience100_factor0.9_notes-alpha=1(low-firing-rates)'
args.load = True
args.load_epoch = 10000
args.load_run = 0
args.data_seed = 4198318570
# args.num_epochs = 50000
# args.notes = 'Full'
# args.batch_size = 'All'
# args.tau_beta = 10
# args.tau_budget = 10000
# args.tau_config = 1
# args.tau_sigma = 1


args.batch_size = 8
args.param_seed = 'TrueInit+MinorPenalty1+Batch'
# args.batch_size = 'All'
# args.param_seed = 'TrueInit+MinorPenalty1+Full'
args.notes = 'alpha=1(low-firing-rates)'
args.scheduler_patience = 100
args.scheduler_factor = 0.9
args.lr = 0.001
args.num_epochs = 50000
args.tau_beta = 1
args.tau_budget = 10
args.tau_config = 1
args.tau_sigma = 1


# outputs_folder = '../../outputs'
print('Start')
output_dir = os.path.join(os.getcwd(), outputs_folder)
# Set the random seed manually for reproducibility.
np.random.seed(args.data_seed)
# if args.param_seed != 'TRUTH':
#     torch.manual_seed(args.param_seed)
# Ground truth data
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("Using CUDA!")
else:
    args.cuda = False
data = DataAnalyzer().initialize(configs=args.n_configs, A=args.A, intensity_mltply=args.intensity_mltply,
                                 intensity_bias=args.intensity_bias)
# Training data
Y_train, factor_access_train = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials), args.cuda)
intensities_train, factor_assignment_train, factor_assignment_onehot_train, neuron_gains_train, trial_offsets_train = load_tensors(data.get_sample_ground_truth(), args.cuda)
# Validation data
Y_test, factor_access_test = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials), args.cuda)
intensities_test, factor_assignment_test, factor_assignment_onehot_test, neuron_gains_test, trial_offsets_test = load_tensors(data.get_sample_ground_truth(), args.cuda)

# initialize the model with ground truth params
num_factors = data.beta.shape[0]
model = LikelihoodELBOModel(data.time, num_factors, args.A, args.n_configs, args.n_trial_samples)
model.init_ground_truth(torch.tensor(data.beta), torch.tensor(data.alpha), torch.tensor(data.theta), torch.tensor(data.pi),
                        torch.tensor(data.config_peak_offsets), torch.tensor(data.trial_peak_offset_covar_ltri))

if args.cuda: model.cuda()
model.eval()
with (torch.no_grad()):
    model.trial_peak_offsets = trial_offsets_train.clone().detach()
    true_ELBO_train, model_trial_offsets_train, model_factor_assignment_train, model_neuron_gains_train = model.evaluate(Y_train, factor_access_train, args.A)

    model.trial_peak_offsets = trial_offsets_test.clone().detach()
    true_ELBO_test, model_trial_offsets_test, model_factor_assignment_test, model_neuron_gains_test = model.evaluate(Y_test, factor_access_test, args.A)

trial_offsets_train = trial_offsets_train.squeeze().permute(1, 0, 2)
trial_offsets_test = trial_offsets_test.squeeze().permute(1, 0, 2)
true_ELBO_train = true_ELBO_train.item()
true_ELBO_test = true_ELBO_test.item()
output_str = (
    f"True ELBO Training: {true_ELBO_train},\n"
    f"True ELBO Test: {true_ELBO_test}\n\n")
patience = args.scheduler_patience//args.eval_interval
if args.load:
    start_epoch = args.load_epoch + 1
    output_dir = os.path.join(output_dir, args.folder_name)
    model_state, optimizer_state, scheduler_state = load_model_checkpoint(os.path.join(output_dir, f'Run_{args.load_run}'), args.load_epoch)
    # model_state, optimizer_state = model_state.state_dict(), optimizer_state.state_dict()
    model.load_state_dict(model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(optimizer_state)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.scheduler_factor,
                                                           patience=patience, threshold=1e-3)
    scheduler.load_state_dict(scheduler_state)
    # output_dir = os.path.join(output_dir, f'Run_{args.load_run+1}')
    output_dir = os.path.join(output_dir, f'Run_{args.load_run}')
else:
    start_epoch = 0
    args.folder_name = (
        f'{args.param_seed}_dataSeed{args.data_seed}_K{args.K}_R{args.n_trials}_A{args.A}_C{args.n_configs}'
        f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauConfig{args.tau_config}_tauSigma{args.tau_sigma}'
        f'_iters{args.num_epochs}_BatchSize{args.batch_size}_lr{args.lr}_patience{args.scheduler_patience}'
        f'_factor{args.scheduler_factor}_notes-{args.notes}')
    output_dir = os.path.join(output_dir, args.folder_name, 'Run_0')
    os.makedirs(output_dir)
    # Initialize the model
    # model.init_from_data(Y=Y_train, factor_access=factor_access_train)
    # model.init_random()
    # model.init_zero()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.scheduler_factor,
                                                           patience=patience, threshold=1e-3)
    create_relevant_files(output_dir, output_str)
    plot_spikes(Y_train.cpu().numpy(), output_dir, data.dt, 'train')
    plot_spikes(Y_test.cpu().numpy(), output_dir, data.dt, 'test')
    plot_intensity_and_latents(data.time, np.exp(data.beta), intensities_train.cpu().numpy(), output_dir)
    # plot_factor_assignments(factor_assignment_onehot_train-model_factor_assignment_train, output_dir, 'Train', -1)
    # plot_factor_assignments(factor_assignment_onehot_test-model_factor_assignment_test, output_dir, 'Test', -1)
    plot_outputs(model.cpu(), args.A, output_dir, 'Train', -1)

# Instantiate the dataset and dataloader
dataset = CustomDataset(Y_train, factor_access_train)
if args.batch_size == 'All':
    args.batch_size = Y_train.shape[0]
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print(f'folder_name: {args.folder_name}')
print(output_str)

if __name__ == "__main__":
    log_likelihoods_batch = []
    losses_batch = []
    log_likelihoods_train = []
    # losses_train = []
    log_likelihoods_test = []
    # losses_test = []
    beta_mses = []
    alpha_mses = []
    theta_mses = []
    pi_mses = []
    config_mses = []
    ltri_mses = []
    clusr_misses_train = []
    clusr_misses_test = []
    gains_train = []
    gains_test = []
    offsets_train = []
    offsets_test = []
    total_time = 0
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.cuda: model.cuda()
        model.train()
        for Y, access in dataloader:
            optimizer.zero_grad()
            likelihood_term, penalty_term = model.forward(Y, access, args.A, args.tau_beta, args.tau_budget, args.tau_config, args.tau_sigma)
            loss = -(likelihood_term + penalty_term)
            loss.backward()
            optimizer.step()
            losses_batch.append((likelihood_term + penalty_term).item())
            log_likelihoods_batch.append(likelihood_term.item())

        if epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                beta_mses.append(F.mse_loss(model.beta, torch.tensor(data.beta).to(model.device)).item())
                alpha_mses.append(F.mse_loss(model.alpha, torch.tensor(data.alpha).to(model.device)).item())
                theta_mses.append(F.mse_loss(model.theta, torch.tensor(data.theta).to(model.device)).item())
                pi_mses.append(F.mse_loss(model.pi, torch.tensor(data.pi).reshape(args.A, -1)[:, 1:].to(model.device)).item())
                config_mses.append(F.mse_loss(model.config_peak_offsets, torch.tensor(data.config_peak_offsets).to(model.device)).item())
                ltri_mses.append(F.mse_loss(model.trial_peak_offset_covar_ltri, torch.tensor(data.trial_peak_offset_covar_ltri).to(model.device)).item())

                likelihood_term_train, model_trial_offsets_train, model_factor_assignment_train, model_neuron_gains_train = model.evaluate(Y_train, factor_access_train, args.A)
                # losses_train.append((likelihood_term_train + penalty_term_train).item())
                log_likelihoods_train.append(likelihood_term_train.item())

                likelihood_term_test, model_trial_offsets_test, model_factor_assignment_test, model_neuron_gains_test = model.evaluate(Y_test, factor_access_test, args.A)
                # losses_test.append((likelihood_term_test + penalty_term_test).item())
                log_likelihoods_test.append(likelihood_term_test.item())
                scheduler.step(likelihood_term_test)
                clusr_misses_train.append(torch.sum(torch.abs(factor_assignment_onehot_train - model_factor_assignment_train)).item())
                clusr_misses_test.append(torch.sum(torch.abs(factor_assignment_onehot_test - model_factor_assignment_test)).item())
                gains_train.append(F.mse_loss(neuron_gains_train, model_neuron_gains_train).item())
                gains_test.append(F.mse_loss(neuron_gains_test, model_neuron_gains_test).item())
                offsets_train.append(F.mse_loss(trial_offsets_train, model_trial_offsets_train).item())
                offsets_test.append(F.mse_loss(trial_offsets_test, model_trial_offsets_test).item())

        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = 0 # losses_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            cur_loss_test = 0 # losses_test[-1]
            with torch.no_grad():
                smoothness_budget_constrained = F.softmax(torch.cat([torch.zeros(1, device=model.device), model.smoothness_budget]), dim=0).cpu().numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"smoothness_budget: {smoothness_budget_constrained.T},\n"
                f"lr: {args.lr:.5f}, scheduler_lr: {scheduler._last_lr[0]:.5f}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model, optimizer, scheduler)
            plot_outputs(model.cpu(), args.A, output_dir, 'Train', epoch)
            is_empty = epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            # write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            # write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            write_losses(beta_mses, 'Test', 'beta_MSE', output_dir, is_empty)
            write_losses(alpha_mses, 'Test', 'alpha_MSE', output_dir, is_empty)
            write_losses(theta_mses, 'Test', 'theta_MSE', output_dir, is_empty)
            write_losses(pi_mses, 'Test', 'pi_MSE', output_dir, is_empty)
            write_losses(config_mses, 'Test', 'configoffset_MSE', output_dir, is_empty)
            write_losses(ltri_mses, 'Test', 'ltri_MSE', output_dir, is_empty)
            write_losses(log_likelihoods_batch, 'Batch', 'Likelihood', output_dir, is_empty)
            write_losses(clusr_misses_train, 'Train', 'clusr_misses', output_dir, is_empty)
            write_losses(clusr_misses_test, 'Test', 'clusr_misses', output_dir, is_empty)
            write_losses(gains_train, 'Train', 'gains_MSE', output_dir, is_empty)
            write_losses(gains_test, 'Test', 'gains_MSE', output_dir, is_empty)
            write_losses(offsets_train, 'Train', 'trialoffsets_MSE', output_dir, is_empty)
            write_losses(offsets_test, 'Test', 'trialoffsets_MSE', output_dir, is_empty)
            write_losses(losses_batch, 'Batch', 'Loss', output_dir, is_empty)
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood')
            # plot_losses(None, output_dir, 'Train', 'Loss', 20)
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood')
            # plot_losses(None, output_dir, 'Test', 'Loss', 1)
            plot_losses(None, output_dir, 'Test', 'beta_MSE')
            plot_losses(None, output_dir, 'Test', 'alpha_MSE')
            plot_losses(None, output_dir, 'Test', 'theta_MSE')
            plot_losses(None, output_dir, 'Test', 'pi_MSE')
            plot_losses(None, output_dir, 'Test', 'configoffset_MSE')
            plot_losses(None, output_dir, 'Test', 'ltri_MSE')
            plot_losses(true_ELBO_train, output_dir, 'Batch', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Batch', 'Loss', 20)
            plot_losses(None, output_dir, 'Train', 'clusr_misses')
            plot_losses(None, output_dir, 'Test', 'clusr_misses')
            plot_losses(None, output_dir, 'Train', 'gains_MSE')
            plot_losses(None, output_dir, 'Test', 'gains_MSE')
            plot_losses(None, output_dir, 'Train', 'trialoffsets_MSE')
            plot_losses(None, output_dir, 'Test', 'trialoffsets_MSE')
            log_likelihoods_batch = []
            losses_batch = []
            log_likelihoods_train = []
            # losses_train = []
            log_likelihoods_test = []
            # losses_test = []
            beta_mses = []
            alpha_mses = []
            theta_mses = []
            pi_mses = []
            config_mses = []
            ltri_mses = []
            clusr_misses_train = []
            clusr_misses_test = []
            gains_train = []
            gains_test = []
            offsets_train = []
            offsets_test = []
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
