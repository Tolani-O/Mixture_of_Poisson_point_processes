import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import create_relevant_files, get_parser, plot_outputs, \
    write_log_and_model, write_losses, plot_epoch_results, write_grad_norms, load_tensors, to_cuda, \
    inv_softplus_torch, preprocess_input_data
import numpy as np
import time
import torch
import torch.nn.functional as F
import threading
outputs_folder = 'outputs'

args = get_parser().parse_args()
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

args.n_trials = 15  # R
args.n_configs = 5  # C
args.K = 100  # K
args.A = 3  # A
args.L = 3  # L
args.n_trial_samples = 10  # Number of samples to generate for each trial

# args.n_trials = 5
# args.n_configs = 3
# args.K = 10
# args.A = 2
# args.n_trial_samples = 1

init = 'Data'
# init = 'Rand'
# init = 'Zero'
# init = 'True'
the_rest = 'zeros'
args.notes = f''
# args.time_warp = False
args.log_interval = 500
args.eval_interval = 500
args.lr = 0.0001
args.num_epochs = 200000
args.temperature = (1, 1000)
args.weights = (99, 1)
# args.tau_beta = 800
# args.tau_config = 500
args.tau_sigma = 1
args.tau_sd = 10000
sd_init = 0.5

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
data = DataAnalyzer().initialize(configs=args.n_configs, A=args.A, L=args.L, intensity_mltply=args.intensity_mltply,
                                 intensity_bias=args.intensity_bias, time_warp=args.time_warp)
# Training data
Y_train, factor_access_train = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials))
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')
_, _, factor_assignment_onehot_train, neuron_gains_train, trial_offsets_train = to_cuda(load_tensors(data.get_sample_ground_truth()),
                                                                                        move_to_cuda=args.cuda)
# Validation data
Y_test, factor_access_test = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials))
_, _, factor_assignment_onehot_test, neuron_gains_test, trial_offsets_test = to_cuda(load_tensors(data.get_sample_ground_truth()),
                                                                                     move_to_cuda=args.cuda)
processed_inputs_train = preprocess_input_data(*to_cuda((Y_train, factor_access_train), move_to_cuda=args.cuda))
processed_inputs_test = preprocess_input_data(*to_cuda((Y_test, factor_access_test), move_to_cuda=args.cuda))
peak1_left_landmarks = data.time[[data.left_landmark1] * args.L]
peak1_right_landmarks = data.time[[data.right_landmark1] * args.L]
peak2_left_landmarks = data.time[[data.left_landmark2] * args.L]
peak2_right_landmarks = data.time[[data.right_landmark2] * args.L]
unique_regions = [f'region{i}' for i in range(args.A)]

# initialize the model with ground truth params
data.load_tensors()
num_factors = data.beta.shape[0]
model = LikelihoodELBOModel(data.time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks,
                            temperature=args.temperature, weights=args.weights)
model.init_ground_truth(beta=data.beta.clone(),
                        alpha=inv_softplus_torch(data.alpha.clone()),
                        config_peak_offsets=data.config_peak_offsets.clone(),
                        trial_peak_offset_covar_ltri=data.trial_peak_offset_covar_ltri.clone(),
                        theta=data.theta.clone(),
                        pi=F.softmax(data.pi.clone().reshape(args.A, -1), dim=1).flatten(),
                        sd_init=1e-5)
model.cuda(move_to_cuda=args.cuda)
with torch.no_grad():
    model.init_ground_truth(trial_peak_offset_proposal_means=trial_offsets_test.clone().squeeze(),
                            W_CKL=factor_assignment_onehot_test.clone(),
                            init='')
    true_ELBO_test = model.forward(processed_inputs_test, update_membership=False, train=False)
    model.init_ground_truth(trial_peak_offset_proposal_means=trial_offsets_train.clone().squeeze(),
                            W_CKL=factor_assignment_onehot_train.clone(),
                            init='')
    true_ELBO_train = model.forward(processed_inputs_train, update_membership=False, train=False)
    likelihood_ground_truth_train = model.log_likelihood(processed_inputs_train, E_step=True)
true_ELBO_train = (1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * true_ELBO_train.item()
true_ELBO_test = (1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * true_ELBO_test.item()
likelihood_ground_truth_train = (1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_ground_truth_train.item()
ltri_matrix = model.ltri_matix()
true_offset_penalty_train = (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_train, ltri_matrix).sum().item()
true_offset_penalty_test = (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_test, ltri_matrix).sum().item()
model.cpu()
args.folder_name = (
    f'seed{args.data_seed}_simulated_{init}Init_K{args.K}_A{args.A}_C{args.n_configs}_L{args.L}'
    f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauConfig{args.tau_config}_tauSigma{args.tau_sigma}_tauSD{args.tau_sd}'
    f'_posterior{args.n_trial_samples}_iters{args.num_epochs}_lr{args.lr}_temp{args.temperature}_weight{args.weights}'
    f'_notes-{args.notes}')
output_dir = os.path.join(os.getcwd(), outputs_folder, args.folder_name, 'Run_0')
os.makedirs(output_dir, exist_ok=True)
plot_outputs(model, unique_regions, output_dir, 'Train', -2)
# Initialize the model
if init == 'True':
    model.init_ground_truth(beta=data.beta.clone(),
                            alpha=inv_softplus_torch(data.alpha.clone()),
                            config_peak_offsets=data.config_peak_offsets.clone(),
                            trial_peak_offset_proposal_means=trial_offsets_train.clone().squeeze().cpu(),
                            trial_peak_offset_covar_ltri=data.trial_peak_offset_covar_ltri.clone(),
                            W_CKL=factor_assignment_onehot_train.clone().cpu(),
                            theta=data.theta.clone(),
                            pi=F.softmax(data.pi.clone().reshape(args.A, -1), dim=1).flatten(),
                            sd_init=1e-5, init=the_rest)
elif init == 'Rand':
    model.init_random()
elif init == 'Zero':
    model.init_zero()
elif init == 'Data':
    model.init_from_data(Y=Y_train, factor_access=processed_inputs_train['neuron_factor_access'].cpu(),
                         sd_init=sd_init, init=the_rest)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
patience = args.scheduler_patience // args.eval_interval
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                       factor=args.scheduler_factor,
                                                       patience=patience, threshold_mode='abs',
                                                       threshold=args.scheduler_threshold)
output_str = (
    f"Using CUDA: {args.cuda}\n"
    f"Num available GPUs: {torch.cuda.device_count()}\n"
    f"peak1_left_landmarks:\n{model.time[model.peak1_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
    f"peak1_right_landmarks:\n{model.time[model.peak1_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
    f"peak2_left_landmarks:\n{model.time[model.peak2_left_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
    f"peak2_right_landmarks:\n{model.time[model.peak2_right_landmarks.reshape(model.n_areas, -1)].numpy()}\n"
    f"True ELBO Training: {true_ELBO_train},\n"
    f"True ELBO Test: {true_ELBO_test},\n"
    f"True Offset Likelihood Training: {true_offset_penalty_train},\n"
    f"True Offset Likelihood Test: {true_offset_penalty_test}\n\n")
params = {
    'peak1_left_landmarks': peak1_left_landmarks.tolist(),
    'peak1_right_landmarks': peak1_right_landmarks.tolist(),
    'peak2_left_landmarks': peak2_left_landmarks.tolist(),
    'peak2_right_landmarks': peak2_right_landmarks.tolist(),
}
create_relevant_files(output_dir, output_str, params=params, ground_truth=True)
plot_outputs(model, unique_regions, output_dir, 'Train', -1)
data.cuda(args.cuda)
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
    log_likelihoods_test = []
    losses_test = []
    beta_mses = []
    alpha_mses = []
    theta_mses = []
    pi_mses = []
    config_mses = []
    ltri_mses = []
    Sigma_mses = []
    proposal_means_mses = []
    ltriLkhd_train = []
    ltriLkhd_test = []
    gains_train = []
    gains_test = []
    batch_grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
    grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
    input_dict = {
        'unique_regions': unique_regions,
        'output_dir': output_dir,
        'batch_grad_norms': list(batch_grad_norms.keys()),
        'grad_norms': list(grad_norms.keys()),
        'likelihood_ground_truth_train': likelihood_ground_truth_train,
        'true_ELBO_train': true_ELBO_train,
        'true_ELBO_test': true_ELBO_test,
        'true_offset_penalty_train': true_offset_penalty_train,
        'true_offset_penalty_test': true_offset_penalty_test,
        'Y': Y_train
    }
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
                likelihood_term_test = model.forward(processed_inputs_test, train=False)
                model_factor_assignment_test, model_neuron_gains_test = model.infer_latent_variables(processed_inputs_test)

                losses_train.append((likelihood_term + penalty_term).item())
                log_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_term.item())
                true_likelihoods_train.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * true_likelihood_term.item())
                losses_test.append((likelihood_term_test + penalty_term).item())
                log_likelihoods_test.append((1 / (args.K * args.n_trials * args.n_configs * model.time.shape[0])) * likelihood_term_test.item())
                epoch_train.append(epoch)
                scheduler.step(log_likelihoods_train[-1])

                non_zero_model_train = torch.nonzero(model_factor_assignment_train)
                non_zero_model_test = torch.nonzero(model_factor_assignment_test)
                non_zero_data_train = torch.nonzero(factor_assignment_onehot_train)
                non_zero_data_test = torch.nonzero(factor_assignment_onehot_test)

                beta_model = model.unnormalized_log_factors()[non_zero_model_train[:, 2]]
                beta_data = data.beta[non_zero_data_train[:, 2]]
                beta_mses.append(F.mse_loss(beta_model, beta_data).item())
                alpha_model = F.softplus(model.alpha[non_zero_model_train[:, 2]])
                alpha_data = data.alpha[non_zero_data_train[:, 2]]
                alpha_mses.append(F.mse_loss(alpha_model, alpha_data).item())
                theta_model = model.theta[non_zero_model_train[:, 2]]
                theta_data = data.theta[non_zero_data_train[:, 2]]
                theta_mses.append(F.mse_loss(theta_model, theta_data).item())
                pi_model = model.pi[non_zero_model_train[:, 2]]
                pi_data = F.softmax(data.pi.reshape(args.A, -1), dim=1).flatten()[non_zero_data_train[:, 2]]
                pi_mses.append(F.mse_loss(pi_model, pi_data).item())
                ltri_matrix = model.ltri_matix()
                ltri_model = ltri_matrix.reshape(ltri_matrix.shape[0], 2, model.n_factors)[:, :,
                             non_zero_model_train[:, 2]]
                ltri_data = data.trial_peak_offset_covar_ltri.reshape(ltri_matrix.shape[0], 2,
                                                                      model.n_factors)[:, :, non_zero_data_train[:, 2]]
                ltri_mses.append(F.mse_loss(ltri_model, ltri_data).item())
                Sigma_model = ltri_matrix @ ltri_matrix.t()
                Sigma_model = Sigma_model.reshape(ltri_matrix.shape[0], 2, model.n_factors)[:, :,
                              non_zero_model_train[:, 2]]
                Sigma_data = data.trial_peak_offset_covar_ltri @ data.trial_peak_offset_covar_ltri.t()
                Sigma_data = Sigma_data.reshape(ltri_matrix.shape[0], 2, model.n_factors)[:, :,
                             non_zero_data_train[:, 2]]
                Sigma_mses.append(F.mse_loss(Sigma_model, Sigma_data).item())
                config_model = model.config_peak_offsets.reshape(model.config_peak_offsets.shape[0], 2,
                                                                 model.n_factors)[non_zero_model_train[:, 0], :,
                               non_zero_model_train[:, 2]]
                config_data = data.config_peak_offsets.reshape(model.config_peak_offsets.shape[0], 2,
                                                               model.n_factors)[non_zero_data_train[:, 0], :,
                              non_zero_data_train[:, 2]]
                config_mses.append(F.mse_loss(config_model, config_data).item())

                gains_train.append(F.mse_loss(neuron_gains_train, model_neuron_gains_train).item())
                gains_test.append(F.mse_loss(neuron_gains_test, model_neuron_gains_test).item())
                trial_offsets_data_train = trial_offsets_train.squeeze().permute(1, 0, 2).reshape(
                    trial_offsets_train.shape[2],
                    trial_offsets_train.shape[1],
                    2, model.n_factors)[non_zero_data_train[:, 0], :, :, non_zero_data_train[:, 2]]
                trial_offsets_proposal_means = model.trial_peak_offset_proposal_means.permute(1, 0, 2).reshape(
                    trial_offsets_train.shape[2],
                    trial_offsets_train.shape[1],
                    2, model.n_factors)[non_zero_data_train[:, 0], :, :, non_zero_data_train[:, 2]]
                proposal_means_mses.append(F.mse_loss(trial_offsets_data_train, trial_offsets_proposal_means).item())
                ltriLkhd_train.append((1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_train, ltri_matrix).sum().item())
                ltriLkhd_test.append((1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_test, ltri_matrix).sum().item())

        if epoch == start_epoch or epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            cur_loss_test = losses_test[-1]
            cur_ltriLkhd_train = ltriLkhd_train[-1]
            cur_ltriLkhd_test = ltriLkhd_test[-1]
            model.cpu()
            with torch.no_grad():
                pi = model.pi.numpy().round(3)
                alpha = F.softplus(model.alpha).numpy().round(3)
                theta = model.theta.numpy().round(3)
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"Loss test: {cur_loss_test:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"ltriLkhd_train: {cur_ltriLkhd_train:.5f}, ltriLkhd_test: {cur_ltriLkhd_test:.5f},\n"
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
            write_losses(log_likelihoods_test, 'test', 'log_likelihoods', output_dir, is_empty)
            write_losses(losses_test, 'test', 'losses', output_dir, is_empty)
            write_losses(beta_mses, 'test', 'beta_MSE', output_dir, is_empty)
            write_losses(alpha_mses, 'test', 'alpha_MSE', output_dir, is_empty)
            write_losses(theta_mses, 'test', 'theta_MSE', output_dir, is_empty)
            write_losses(pi_mses, 'test', 'pi_MSE', output_dir, is_empty)
            write_losses(config_mses, 'test', 'configoffset_MSE', output_dir, is_empty)
            write_losses(ltri_mses, 'test', 'ltri_MSE', output_dir, is_empty)
            write_losses(Sigma_mses, 'test', 'Sigma_MSE', output_dir, is_empty)
            write_losses(proposal_means_mses, 'test', 'proposal_means_MSE', output_dir, is_empty)
            write_losses(gains_train, 'train', 'gains_MSE', output_dir, is_empty)
            write_losses(gains_test, 'test', 'gains_MSE', output_dir, is_empty)
            write_losses(ltriLkhd_train, 'train', 'ltriLkhd', output_dir, is_empty)
            write_losses(ltriLkhd_test, 'test', 'ltriLkhd', output_dir, is_empty)

            input_dict['epoch'] = epoch
            input_dict['model'] = model
            plot_thread = threading.Thread(target=plot_epoch_results, args=(input_dict, True))
            plot_thread.start()

            true_likelihoods_train = []
            log_likelihoods_batch = []
            losses_batch = []
            epoch_batch = []
            log_likelihoods_train = []
            losses_train = []
            epoch_train = []
            log_likelihoods_test = []
            losses_test = []
            beta_mses = []
            alpha_mses = []
            theta_mses = []
            pi_mses = []
            config_mses = []
            ltri_mses = []
            Sigma_mses = []
            proposal_means_mses = []
            ltriLkhd_train = []
            ltriLkhd_test = []
            gains_train = []
            gains_test = []
            batch_grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
            grad_norms = {name: [] for name, param in model.named_parameters() if param.requires_grad}
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
