import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.Allen_data_torch import EcephysAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import initialize_clusters, create_relevant_files, get_parser, plot_outputs, \
    plot_initial_clusters, write_log_and_model, write_losses, plot_losses, CustomDataset, load_tensors
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

args = get_parser().parse_args()
args.data_seed = np.random.randint(0, 2 ** 32 - 1)
outputs_folder = 'outputs'

init = 'Data'
the_rest = 'zeros'
args.batch_size = 'All'
args.param_seed = f'Real_{init}Init'
args.notes = f'var landmarks spread aligned lr 1e-4'
args.scheduler_patience = 80000 #2000
args.scheduler_threshold = 1e-10 #0.1
args.scheduler_factor = 0.9
args.lr = 0.0001
args.num_epochs = 100000
args.tau_beta = 8000
args.tau_config = 10 # Number of samples to generate for each trial
args.tau_sigma = 1
args.tau_sd = 50
sd_init = 0.5
# args.cuda = False
args.n_trial_samples = 10
peak1_left_landmarks = [0.03, 0.03, 0.03]
peak1_right_landmarks = [0.11, 0.14, 0.13]
peak2_left_landmarks = [0.17, 0.18, 0.18]
peak2_right_landmarks = [0.31, 0.32, 0.30]
dt = 0.002

regions = ['VISp', 'VISl', 'VISal']
conditions = None
regions = ['VISp', 'VISl']
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
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')
Y_train, factor_access_train = load_tensors((Y_train, factor_access_train), args.cuda)

args.K, T, args.n_trials, args.n_configs = Y_train.shape
num_factors = factor_access_train.shape[1]
args.A = int(num_factors/args.L)
if not os.path.exists(os.path.join(data.output_dir, folder_name, f'cluster_initialization.pkl')):
    initialize_clusters(Y_train.cpu(), factor_access_train.cpu(), args.L, args.A, os.path.join(data.output_dir, folder_name), n_jobs=15, bandwidth=4)
    plot_initial_clusters(data.output_dir, folder_name, args.L)
    sys.exit()
model = LikelihoodELBOModel(bin_time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks)
# Initialize the model
model.init_from_data(Y=Y_train, factor_access=factor_access_train, sd_init=sd_init, cluster_dir=os.path.join(data.output_dir, folder_name), init=the_rest)

if args.cuda: model.cuda()
model.eval()
with (torch.no_grad()):
    model.generate_trial_peak_offset_samples()
    _, _, _, effective_sample_size_train, trial_peak_offsets_train = model.evaluate(Y_train, factor_access_train)

output_str = (f"Using CUDA: {args.cuda}\n"
              f"Num available GPUs: {torch.cuda.device_count()}\n"
              f"peak1_left_landmarks:\n{model.time[model.peak1_left_landmarks.reshape(model.n_areas, -1)]}\n"
              f"peak1_right_landmarks:\n{model.time[model.peak1_right_landmarks.reshape(model.n_areas, -1)]}\n"
              f"peak2_left_landmarks:\n{model.time[model.peak2_left_landmarks.reshape(model.n_areas, -1)]}\n"
              f"peak2_right_landmarks:\n{model.time[model.peak2_right_landmarks.reshape(model.n_areas, -1)]}\n")
patience = args.scheduler_patience//args.eval_interval
start_epoch = 0
args.folder_name = (
    f'dataSeed{args.data_seed}_{args.param_seed}_K{args.K}_A{args.A}_C{args.n_configs}'
    f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauConfig{args.tau_config}_tauSigma{args.tau_sigma}_tauSD{args.tau_sd}'
    f'_IS{args.n_trial_samples}_iters{args.num_epochs}_BatchSize{args.batch_size}_lr{args.lr}_patience{args.scheduler_patience}'
    f'_factor{args.scheduler_factor}_threshold{args.scheduler_threshold}_notes-{args.notes}')
output_dir = os.path.join(output_dir, args.folder_name, 'Run_0')
os.makedirs(output_dir, exist_ok=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                       factor=args.scheduler_factor,
                                                       patience=patience, threshold_mode='abs',
                                                       threshold=args.scheduler_threshold)
create_relevant_files(output_dir, output_str)
plot_outputs(model.cpu(), factor_access_train.permute(2, 0, 1).cpu(), unique_regions, output_dir, 'Train', -1,
             effective_sample_size_train.cpu(), None,
             trial_peak_offsets_train.permute(1,0,2).cpu(), None)

# Instantiate the dataset and dataloader
dataset = CustomDataset(Y_train, factor_access_train)
if args.batch_size == 'All':
    args.batch_size = Y_train.shape[0]
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print(f'folder_name: {args.folder_name}\n\n')
print(output_str)
# torch.autograd.set_detect_anomaly(True)

def train_gradient():
    for Y, access in dataloader:
        optimizer.zero_grad()
        likelihood_term, penalty_term = model.forward(Y, access, args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
        loss = -(likelihood_term + penalty_term)
        loss.backward()
        optimizer.step()
        losses_batch.append((likelihood_term + penalty_term).item())
        log_likelihoods_batch.append(likelihood_term.item())
        torch.cuda.empty_cache()


if __name__ == "__main__":
    log_likelihoods_batch = []
    losses_batch = []
    log_likelihoods_train = []
    losses_train = []
    total_time = 0
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.cuda: model.cuda()
        model.train()
        train_gradient()
        if epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
                model.generate_trial_peak_offset_samples()
                likelihood_term_train, model_factor_assignment_train, model_neuron_gains_train, effective_sample_size_train, model_trial_offsets_train = model.evaluate(
                    Y_train, factor_access_train)
                losses_train.append(((1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_train +
                                     (1 / (args.n_trials * args.n_configs)) * penalty_term).item())
                log_likelihoods_train.append(
                    (1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_train.item())
                scheduler.step(log_likelihoods_train[-1])

                non_zero_model_train = torch.nonzero(model_factor_assignment_train)
                trial_offsets_model_train = model_trial_offsets_train.reshape(model_trial_offsets_train.shape[0],
                                                                              model_trial_offsets_train.shape[1],
                                                                              2, model.n_factors)[
                                            non_zero_model_train[:, 0], :, :, non_zero_model_train[:, 2]]

        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            with torch.no_grad():
                pi = model.pi_value(factor_access_train.permute(2, 0, 1)).cpu().numpy().round(3)
                alpha = F.softplus(model.alpha).cpu().numpy().round(3)
                theta = model.theta_value().cpu().numpy().round(3)
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
            write_log_and_model(output_str, output_dir, epoch, model.cpu(), optimizer, scheduler)
            plot_outputs(model.cpu(), factor_access_train.permute(2, 0, 1).cpu(), unique_regions, output_dir, 'Train',
                         epoch, effective_sample_size_train.cpu(), None,
                         model_trial_offsets_train.permute(1,0,2).cpu(), None)
            is_empty = epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods_batch, 'Batch', 'Likelihood', output_dir, is_empty)
            write_losses(losses_batch, 'Batch', 'Loss', output_dir, is_empty)
            plot_losses(None, output_dir, 'Train', 'Likelihood', 10)
            plot_losses(None, output_dir, 'Train', 'Loss', 10)
            plot_losses(None, output_dir, 'Batch', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Batch', 'Loss', 20)
            log_likelihoods_batch = []
            losses_batch = []
            log_likelihoods_train = []
            losses_train = []
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
