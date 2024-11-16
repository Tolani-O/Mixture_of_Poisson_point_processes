import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.Allen_data_torch import load_sample
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import parse_folder_name, load_model_checkpoint, create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_epoch_results, write_grad_norms, \
    load_tensors, to_cuda, preprocess_input_data, compute_uncertainty
import numpy as np
import time
import torch
import torch.nn.functional as F
import threading
from ast import literal_eval
outputs_folder = 'outputs'

args = get_parser().parse_args()
parser_key = ['seed', 'A', 'L', 'tauBeta', 'tauConfig', 'tauSigma', 'tauSD', 'posterior', 'iters', 'lr', 'temp', 'weight', 'notes']
args.folder_name = ('seed2997063451_Real_DataInit_K222_A3_C40_L5_R15_tauBeta800_tauConfig500_tauSigma1_tauSD10000_'
                    'posterior7_iters200000_lr0.0001_temp(1, 1000)_weight(10, 1)_notes-masking 10 the_rest zeros')
parser_dict = parse_folder_name(args.folder_name, parser_key, outputs_folder, args.load_run)

args.data_seed = int(parser_dict['seed'])
args.A = int(parser_dict['A'])  # A
args.L = int(parser_dict['L'])  # L
args.notes = parser_dict['notes']
args.log_interval = 500
args.eval_interval = 500
args.lr = float(parser_dict['lr'])
args.temperature = literal_eval(parser_dict['temp'])
args.weights = literal_eval(parser_dict['weight'])
if args.num_epochs >= 0:
    args.num_epochs = int(parser_dict['iters'])
args.tau_beta = float(parser_dict['tauBeta'])
args.tau_config = float(parser_dict['tauConfig'])
args.tau_sigma = float(parser_dict['tauSigma'])
args.tau_sd = float(parser_dict['tauSD'])
args.n_trial_samples = int(parser_dict['posterior'])  # Number of samples to generate for each trial
dt = 0.002

if args.eval_interval > args.log_interval:
    args.log_interval = args.eval_interval
# outputs_folder = '../../outputs'
print('Start\n\n')
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

# Training data
region_ct = args.A
folder_path = os.path.join(os.getcwd(), outputs_folder, 'metadata')
folder_name = f'sample_data_{region_ct}-regions_{args.L}-factors_{dt}_dt'
Y_train, bin_time, factor_access_train, unique_regions = load_sample(folder_path, folder_name)
if Y_train is None:
    raise ValueError("No training data found")
processed_inputs_train = preprocess_input_data(*to_cuda(load_tensors((Y_train, factor_access_train)),
                                                        move_to_cuda=args.cuda), mask_threshold=5)
peak1_left_landmarks = parser_dict['peak1_left_landmarks']
peak1_right_landmarks = parser_dict['peak1_right_landmarks']
peak2_left_landmarks = parser_dict['peak2_left_landmarks']
peak2_right_landmarks = parser_dict['peak2_right_landmarks']
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')

args.K, T, args.n_trials, args.n_configs = Y_train.shape
num_factors = factor_access_train.shape[-1]
args.A = int(num_factors / args.L)
model = LikelihoodELBOModel(bin_time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                            temperature=args.temperature, weights=args.weights)
output_dir = os.path.join(os.getcwd(), outputs_folder, args.folder_name, f'Run_{args.load_run + 1}')
os.makedirs(output_dir, exist_ok=True)
# Load the model
load_dir = os.path.join(os.getcwd(), outputs_folder, args.folder_name, f'Run_{args.load_run}')
model_state, optimizer_state, scheduler_state, W_CKL, a_CKL, theta, pi, args.load_epoch = load_model_checkpoint(load_dir, args.load_epoch)
model.init_zero()
model.load_state_dict(model_state)
model.W_CKL, model.a_CKL, model.theta, model.pi = W_CKL, a_CKL, theta, pi
if args.num_epochs < 0:
    model.cuda(move_to_cuda=args.cuda)
    se_dict = compute_uncertainty(model, processed_inputs_train, output_dir, args.load_epoch)
    model.cpu()
    plot_outputs(model, unique_regions, output_dir, 'Train', args.load_epoch, se_dict)
    interpret_results(model, processed_inputs_train, 'output_dir', -2)
    sys.exit()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer.load_state_dict(optimizer_state)
if args.cuda:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to('cuda')
patience = args.scheduler_patience // args.eval_interval
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                       factor=args.scheduler_factor,
                                                       patience=patience, threshold_mode='abs',
                                                       threshold=args.scheduler_threshold)
scheduler.load_state_dict(scheduler_state)
output_str = (f"Using CUDA: {args.cuda}\n"
              f"Num available GPUs: {torch.cuda.device_count()}\n"
              f"peak1_left_landmarks:\n{model.time[model.peak1_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak1_right_landmarks:\n{model.time[model.peak1_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_left_landmarks:\n{model.time[model.peak2_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
              f"peak2_right_landmarks:\n{model.time[model.peak2_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n\n")
params = {
    'peak1_left_landmarks': peak1_left_landmarks,
    'peak1_right_landmarks': peak1_right_landmarks,
    'peak2_left_landmarks': peak2_left_landmarks,
    'peak2_right_landmarks': peak2_right_landmarks,
}
create_relevant_files(output_dir, output_str, params=params)
plot_outputs(model, unique_regions, output_dir, 'Train', -1)
print(f'folder_name: {args.folder_name}\n\n')
print(output_str)

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    true_likelihoods_train = []
    log_likelihoods_batch = []
    losses_batch = []
    epoch_batch = []
    log_likelihoods_train = []
    losses_train = []
    epoch_train = []
    batch_grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
    grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
    input_dict = {
        'unique_regions': unique_regions,
        'output_dir': output_dir,
        'batch_grad_norms': list(batch_grad_norms.keys()),
        'grad_norms': list(grad_norms.keys()),
        'likelihood_ground_truth_train': None,
        'true_ELBO_train': None,
        'Y': Y_train,
        'model_params': {
            'time': bin_time,
            'n_factors': num_factors,
            'n_areas': args.A,
            'n_configs': args.n_configs,
            'n_trials': args.n_trials,
            'n_trial_samples': args.n_trial_samples,
            'peak1_left_landmarks': peak1_left_landmarks,
            'peak1_right_landmarks': peak1_right_landmarks,
            'peak2_left_landmarks': peak2_left_landmarks,
            'peak2_right_landmarks': peak2_right_landmarks,
            'temperature': args.temperature,
            'weights': args.weights
        }
    }
    total_time = 0
    start_time = time.time()
    batch_ct = 0
    start_epoch = args.load_epoch + 1
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
        model_named_parameters = dict(model.named_parameters())
        [batch_grad_norms[name].append(model_named_parameters[name].grad.norm().item()) for name in batch_grad_norms.keys()]
        batch_ct += 1
        torch.cuda.empty_cache()
        if epoch == start_epoch or epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            with torch.no_grad():
                [grad_norms[name].append(model_named_parameters[name].grad.norm().item()) for name in grad_norms.keys()]
                penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
                likelihood_term = model.forward(processed_inputs_train, train=False)
                true_likelihood_term = model.log_likelihood(processed_inputs_train)
                model_factor_assignment_train, model_neuron_gains_train = model.infer_latent_variables(processed_inputs_train)
                losses_train.append((likelihood_term + penalty_term).item())
                log_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_term.item())
                true_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * true_likelihood_term.item())
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
            is_empty = epoch == start_epoch
            write_grad_norms(batch_grad_norms, 'batch', output_dir, is_empty)
            write_grad_norms(grad_norms, 'train', output_dir, is_empty)
            write_losses(true_likelihoods_train, 'train', 'true_log_likelihoods', output_dir, is_empty)
            write_losses(log_likelihoods_train, 'train', 'log_likelihoods', output_dir, is_empty)
            write_losses(losses_train, 'train', 'losses', output_dir, is_empty)
            write_losses(epoch_train, 'train', 'epoch', output_dir, is_empty)
            write_losses(log_likelihoods_batch, 'batch', 'log_likelihoods', output_dir, is_empty)
            write_losses(losses_batch, 'batch', 'losses', output_dir, is_empty)
            write_losses(epoch_batch, 'batch', 'epoch', output_dir, is_empty)

            input_dict['epoch'] = epoch
            plot_thread = threading.Thread(target=plot_epoch_results, args=(input_dict, False))
            plot_thread.start()

            true_likelihoods_train = []
            log_likelihoods_batch = []
            losses_batch = []
            epoch_batch = []
            log_likelihoods_train = []
            losses_train = []
            epoch_train = []
            batch_grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
            grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
