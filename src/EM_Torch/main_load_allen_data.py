import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.Allen_data_torch import EcephysAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import initialize_clusters, create_relevant_files, get_parser, plot_outputs, \
    plot_initial_clusters, write_log_and_model, write_losses, plot_losses, CustomDataset, load_tensors, to_cuda, to_cpu, \
    preprocess_input_data
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
outputs_folder = 'outputs'

args = get_parser().parse_args()
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

init = 'Data'
the_rest = 'zeros'
args.notes = f'dt1x10000 assgn lt 15 units'
args.log_interval = 500
args.eval_interval = 500
args.lr = 0.0001
args.num_epochs = 200000
args.temperature = (1, 1000)
args.weights = (10, 1)
# args.tau_beta = 800
# args.tau_config = 500
args.tau_sigma = 1
args.tau_sd = 10000
sd_init = 0.5
# args.cuda = False
args.init_with_DTW = False
args.n_trial_samples = 10  # Number of samples to generate for each trial
peak1_left_landmarks = [0.03, 0.03, 0.03]
peak1_right_landmarks = [0.11, 0.14, 0.13]
peak2_left_landmarks = [0.17, 0.18, 0.18]
peak2_right_landmarks = [0.31, 0.32, 0.30]
dt = 0.002

regions = None
conditions = None
regions = ['VISp', 'VISl', 'VISal']
# conditions = [246, 251]

if args.eval_interval > args.log_interval:
    args.log_interval = args.eval_interval
# outputs_folder = '../../outputs'
print('Start\n\n')
output_dir = os.path.join(os.getcwd(), outputs_folder)
# Set the random seed manually for reproducibility.
np.random.seed(args.data_seed)
# Ground truth data
if torch.cuda.is_available():
    print("Number of available GPUs:", torch.cuda.device_count())
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("Using CUDA!")
else:
    args.cuda = False

data = EcephysAnalyzer(structure_list=regions, spike_train_start_offset=0, spike_train_end=0.35, dt=dt)
# Training data
region_ct = len(regions) if regions is not None else 7
args.L = len(peak1_left_landmarks)
folder_name = f'sample_data_{region_ct}-regions_{args.L}-factors_{dt}_dt'
Y_train, bin_time, factor_access_train, unique_regions = data.load_sample(folder_name)
if Y_train is None:
    data.initialize()
    data.plot_presentations_times(folder_name)
    data.plot_spike_times(folder_name)
    data.plot_spike_counts(folder_name)
    Y_train, bin_time, factor_access_train, unique_regions = data.sample_data(conditions=conditions, num_factors=args.L)
    data.save_sample(Y_train, bin_time, factor_access_train, unique_regions, folder_name)
Y_train, factor_access_train = load_tensors((Y_train, factor_access_train))
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')

args.K, T, args.n_trials, args.n_configs = Y_train.shape
num_factors = factor_access_train.shape[-1]
args.A = int(num_factors/args.L)
if args.init_with_DTW:
    cluster_dir = os.path.join(data.output_dir, folder_name)
    if not os.path.exists(os.path.join(data.output_dir, folder_name, f'cluster_initialization.pkl')):
        initialize_clusters(Y_train, factor_access_train, args.L, args.A, cluster_dir, n_jobs=15, bandwidth=4)
        plot_initial_clusters(data.output_dir, folder_name, args.L)
        # sys.exit()
else:
    cluster_dir = None
model = LikelihoodELBOModel(bin_time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                            temperature=args.temperature, weights=args.weights)
# Initialize the model
processed_inputs_train = preprocess_input_data(*to_cuda((Y_train, factor_access_train), move_to_cuda=args.cuda),
                                               mask_threshold=5)
model.init_from_data(Y=Y_train, factor_access=processed_inputs_train['neuron_factor_access'].cpu(),
                     sd_init=sd_init, cluster_dir=cluster_dir, init=the_rest)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
patience = args.scheduler_patience//args.eval_interval
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                       factor=args.scheduler_factor,
                                                       patience=patience, threshold_mode='abs',
                                                       threshold=args.scheduler_threshold)
args.folder_name = (
    f'seed{args.data_seed}_Real_{init}Init_K{args.K}_A{args.A}_C{args.n_configs}_L{args.L}'
    f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauConfig{args.tau_config}_tauSigma{args.tau_sigma}_tauSD{args.tau_sd}'
    f'_posterior{args.n_trial_samples}_iters{args.num_epochs}_lr{args.lr}_temp{args.temperature}_weight{args.weights}'
    f'_notes-{args.notes}')
output_dir = os.path.join(output_dir, args.folder_name, 'Run_0')
os.makedirs(output_dir, exist_ok=True)
output_str = (f"Using CUDA: {args.cuda}\n"
              f"Num available GPUs: {torch.cuda.device_count()}\n"
              f"peak1_left_landmarks:\n{model.time[model.peak1_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak1_right_landmarks:\n{model.time[model.peak1_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_left_landmarks:\n{model.time[model.peak2_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_right_landmarks:\n{model.time[model.peak2_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n\n")
create_relevant_files(output_dir, output_str)
plot_outputs(model, unique_regions, output_dir, 'Train', -1)

# Instantiate the dataset and dataloader
print(f'folder_name: {args.folder_name}\n\n')
print(output_str)

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    log_likelihoods_batch = []
    losses_batch = []
    epoch_batch = []
    log_likelihoods_train = []
    losses_train = []
    epoch_train = []
    total_time = 0
    start_time = time.time()
    batch_ct = 0
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.cuda(move_to_cuda=args.cuda)
        optimizer.zero_grad()
        likelihood_term = model.forward(processed_inputs_train)
        penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
        loss = -(likelihood_term + penalty_term)
        loss.backward()
        optimizer.step()
        losses_batch.append((likelihood_term + penalty_term).item())
        log_likelihoods_batch.append(likelihood_term.item())
        epoch_batch.append(batch_ct)
        batch_ct += 1
        torch.cuda.empty_cache()
        if epoch == start_epoch or epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            with torch.no_grad():
                penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
                likelihood_term = model.forward(processed_inputs_train, train=False)
                model_factor_assignment_train, model_neuron_gains_train = model.infer_latent_variables()
                losses_train.append((likelihood_term + penalty_term).item())
                log_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_term.item())
                epoch_train.append(epoch)
                scheduler.step(log_likelihoods_train[-1])
        if epoch == start_epoch or epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            model.cpu()
            with torch.no_grad():
                pi = model.pi.numpy().round(3)
                alpha = F.softplus(model.alpha).numpy().round(3)
                theta = model.theta.numpy().round(3)
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f},\n"
                f"Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"pi:\n{pi.T.reshape(model.n_areas, -1)},\n"
                f"alpha:\n{alpha.reshape(model.n_areas, -1)},\n"
                f"theta:\n{theta.reshape(model.n_areas, -1)},\n"
                f"lr: {args.lr:.5f}, scheduler_lr: {scheduler._last_lr[0]:.5f},\n"
                f"dataSeed: {args.data_seed},\n"
                f"{args.notes}\n\n")
            write_log_and_model(output_str, output_dir, epoch, model, optimizer, scheduler)
            plot_outputs(model, unique_regions, output_dir, 'Train', epoch)
            is_empty = epoch == start_epoch
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(epoch_train, 'Train', 'Epoch', output_dir, is_empty)
            write_losses(log_likelihoods_batch, 'Batch', 'Likelihood', output_dir, is_empty)
            write_losses(losses_batch, 'Batch', 'Loss', output_dir, is_empty)
            write_losses(epoch_batch, 'Batch', 'Epoch', output_dir, is_empty)
            plot_losses(None, output_dir, 'Train', 'Likelihood', 10)
            plot_losses(None, output_dir, 'Train', 'Loss', 10)
            plot_losses(None, output_dir, 'Batch', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Batch', 'Loss', 20)
            log_likelihoods_batch = []
            losses_batch = []
            epoch_batch = []
            log_likelihoods_train = []
            losses_train = []
            epoch_train = []
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
