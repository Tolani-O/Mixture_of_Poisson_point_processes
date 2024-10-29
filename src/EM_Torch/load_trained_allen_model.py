import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.Allen_data_torch import EcephysAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import parse_folder_name, load_model_checkpoint, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_losses, CustomDataset, load_tensors, to_cuda, to_cpu
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
outputs_folder = 'outputs'

args = get_parser().parse_args()
parser_key = ['dataSeed', 'tauBeta', 'tauConfig', 'tauSigma', 'tauSD', 'IS', 'iters', 'BatchSize', 'lr', 'patience', 'factor', 'threshold', 'temp', 'notes']
args.folder_name = 'dataSeed1900327208_Real_DataInit_K222_A3_C40_R15_tauBeta1000.0_tauConfig6000.0_tauSigma1_tauSD10000_IS10_iters200000_BatchSizeAll_lr0.0001_patience80000_factor0.9_threshold1e-10_temp1.0_notes-Learn all'
parser_dict = parse_folder_name(args.folder_name, parser_key)

args.data_seed = int(parser_dict['dataSeed'])
args.batch_size = parser_dict['BatchSize']
args.notes = parser_dict['notes']
args.log_interval = 500
args.eval_interval = 500
args.scheduler_patience = int(parser_dict['patience'])
args.scheduler_threshold = float(parser_dict['threshold'])
args.scheduler_factor = float(parser_dict['factor'])
args.lr = float(parser_dict['lr'])
args.temperature = float(parser_dict['temp'])
args.num_epochs = int(parser_dict['iters'])
args.tau_beta = float(parser_dict['tauBeta'])
args.tau_config = float(parser_dict['tauConfig'])
args.tau_sigma = float(parser_dict['tauSigma'])
args.tau_sd = float(parser_dict['tauSD'])
# args.load_epoch = 199999
# args.load_run = 0
args.n_trial_samples = int(parser_dict['IS'])  # Number of samples to generate for each trial
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
    raise ValueError("No training data found")
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')
Y_train, factor_access_train = load_tensors((Y_train, factor_access_train))

args.K, T, args.n_trials, args.n_configs = Y_train.shape
num_factors = factor_access_train.shape[-1]
args.A = int(num_factors / args.L)
model = LikelihoodELBOModel(bin_time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                            temperature=args.temperature)
# Load the model
load_dir = os.path.join(output_dir, args.folder_name, f'Run_{args.load_run}')
model_state, optimizer_state, scheduler_state, W_CKL, a_CKL, theta, pi, args.load_epoch = load_model_checkpoint(load_dir, args.load_epoch)
model.init_zero()
model.load_state_dict(model_state)
model.W_CKL, model.a_CKL, model.theta, model.pi = W_CKL, a_CKL, theta, pi
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer.load_state_dict(optimizer_state)
patience = args.scheduler_patience // args.eval_interval
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                       factor=args.scheduler_factor,
                                                       patience=patience, threshold_mode='abs',
                                                       threshold=args.scheduler_threshold)
scheduler.load_state_dict(scheduler_state)
output_dir = os.path.join(output_dir, args.folder_name, f'Run_{args.load_run + 1}')
os.makedirs(output_dir, exist_ok=True)
output_str = (f"Using CUDA: {args.cuda}\n"
              f"Num available GPUs: {torch.cuda.device_count()}\n"
              f"peak1_left_landmarks:\n{model.time[model.peak1_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak1_right_landmarks:\n{model.time[model.peak1_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_left_landmarks:\n{model.time[model.peak2_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_right_landmarks:\n{model.time[model.peak2_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n")
create_relevant_files(output_dir, output_str)
plot_outputs(model, unique_regions, output_dir, 'Train', -1)

# Instantiate the dataset and dataloader
if args.cuda:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to('cuda')
Y_train, factor_access_train = to_cuda((Y_train, factor_access_train), move_to_cuda=args.cuda)
dataset = CustomDataset(Y_train, factor_access_train)
if args.batch_size == 'All':
    args.batch_size = Y_train.shape[0]
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
print(f'folder_name: {args.folder_name}\n\n')
print(output_str)

# torch.autograd.set_detect_anomaly(True)
def train_gradient(batch_ct):
    for Y, access in dataloader:
        # K x C x L --> C x K x L
        access = access.permute(1, 0, 2)
        optimizer.zero_grad()
        likelihood_term = model.forward(Y, access)
        penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
        loss = -(likelihood_term + penalty_term)
        loss.backward()
        optimizer.step()
        losses_batch.append((likelihood_term + penalty_term).item())
        log_likelihoods_batch.append(likelihood_term.item())
        epoch_batch.append(batch_ct)
        batch_ct += 1
        torch.cuda.empty_cache()
        return batch_ct


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
    start_epoch = args.load_epoch + 1
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.cuda(move_to_cuda=args.cuda)
        batch_ct = train_gradient(batch_ct)
        if epoch == start_epoch or epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            with torch.no_grad():
                factor_access_train = to_cuda([factor_access_train], move_to_cuda=args.cuda)[0]
                penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
                likelihood_term_train = model.forward(Y_train, factor_access_train, train=False)
                model_factor_assignment_train, model_neuron_gains_train = model.infer_latent_variables()
                losses_train.append((likelihood_term_train + penalty_term).item())
                log_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_term_train.item())
                epoch_train.append(epoch)
                scheduler.step(log_likelihoods_train[-1])
        if epoch == start_epoch or epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            model.cpu()
            factor_access_train = to_cpu([factor_access_train])[0]
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
