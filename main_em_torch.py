import os
import sys

sys.path.append(os.path.abspath('.'))
from src.EM_Torch.simulate_data_multitrial import DataAnalyzer
from src.EM_Torch.LikelihoodELBOModel import LikelihoodELBOModel
from src.EM_Torch.general_functions import initialize_clusters, create_relevant_files, get_parser, plot_outputs, \
    plot_initial_clusters, write_log_and_model, write_losses, plot_losses, CustomDataset, load_tensors, \
    inv_softplus_torch
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

args.n_trials = 15  # R
args.n_configs = 40  # C
args.K = 100  # K
# args.intensity_bias = 10
args.A = 7  # A

args.n_trials = 5
args.n_configs = 3
args.K = 6
args.A = 2

init = 'Data'
# init = 'Rand'
# init = 'Zero'
# init = 'True'
the_rest = 'zeros'
# init = 'TrueBeta'
# init = 'TrueAndRandBeta'
# init = 'DataAndZeroBeta'
args.batch_size = 'All'
args.param_seed = f'Real_{init}Init'
args.notes = f'var landmarks spread aligned lr 1e-4 changed hyperparameters'
args.log_interval = 500
args.eval_interval = 500
args.scheduler_patience = 80000  # 2000
args.scheduler_threshold = 1e-10  # 0.1
args.scheduler_factor = 0.9
args.lr = 0.0001
args.num_epochs = 200000
args.tau_beta = 1000
args.tau_config = 5
args.tau_sigma = 0.01
args.tau_sd = 10
sd_init = 0.5
args.n_trial_samples = 4 # 10  # Number of samples to generate for each trial
peak1_left_landmarks = [0.03, 0.03, 0.03]
peak1_right_landmarks = [0.11, 0.14, 0.13]
peak2_left_landmarks = [0.17, 0.18, 0.18]
peak2_right_landmarks = [0.31, 0.32, 0.30]

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
data = DataAnalyzer().initialize(configs=args.n_configs, A=args.A, intensity_mltply=args.intensity_mltply,
                                 intensity_bias=args.intensity_bias)
# Training data
Y_train, factor_access_train = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials), is_numpy=True)
print(f'Y_train shape: {Y_train.shape}, factor_access_train shape: {factor_access_train.shape}')
_, _, factor_assignment_onehot_train, neuron_gains_train, trial_offsets_train = load_tensors(
    data.get_sample_ground_truth(), is_numpy=True, to_cuda=args.cuda)
# Validation data
Y_test, factor_access_test = load_tensors(data.sample_data(K=args.K, A=args.A, n_trials=args.n_trials), is_numpy=True)
_, _, factor_assignment_onehot_test, neuron_gains_test, trial_offsets_test = load_tensors(
    data.get_sample_ground_truth(), is_numpy=True, to_cuda=args.cuda)
unique_regions = [f'region{i}' for i in range(args.A)]

# initialize the model with ground truth params
data.load_tensors(args.cuda)
num_factors = data.beta.shape[0]
model = LikelihoodELBOModel(data.time, num_factors, args.A, args.n_configs, args.n_trials, args.n_trial_samples,
                            peak1_left_landmarks, peak1_right_landmarks, peak2_left_landmarks, peak2_right_landmarks)
model.init_zero()
if args.cuda:
    model.cuda()
    Y_test, factor_access_test, Y_train, factor_access_train = load_tensors((Y_test, factor_access_test, Y_train, factor_access_train), to_cuda=args.cuda)
model.eval()
with (torch.no_grad()):
    model.init_ground_truth(beta=data.beta[:, 1:],
                            alpha=inv_softplus_torch(data.alpha),
                            config_peak_offsets=data.config_peak_offsets,
                            trial_peak_offset_covar_ltri=data.trial_peak_offset_covar_ltri,
                            theta=data.theta,
                            pi=F.softmax(data.pi.reshape(args.A, -1), dim=1).flatten(),
                            trial_peak_offset_proposal_sds=sd_init * torch.ones(2 * data.beta.shape[0], dtype=torch.float64, device=model.device),
                            init='')

    model.init_ground_truth(trial_peak_offset_proposal_means=trial_offsets_test.squeeze(), init='')
    true_ELBO_test = model.forward(Y_test, factor_access_test)
    model.init_ground_truth(trial_peak_offset_proposal_means=trial_offsets_train.squeeze(), init='')
    model.W_CKL = None
    true_ELBO_train = model.forward(Y_train, factor_access_train)
    model.cpu()
    Y_train, factor_access_train = load_tensors((Y_train, factor_access_train))

true_ELBO_train = (1 / (args.K * args.n_trials * args.n_configs)) * true_ELBO_train.item()
true_ELBO_test = (1 / (args.K * args.n_trials * args.n_configs)) * true_ELBO_test.item()
true_offset_penalty_train = (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_train,
                                                                                                model.ltri_matix()).sum().item()
true_offset_penalty_test = (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_test,
                                                                                               model.ltri_matix()).sum().item()
args.folder_name = (
    f'dataSeed{args.data_seed}_{args.param_seed}_K{args.K}_A{args.A}_C{args.n_configs}'
    f'_R{args.n_trials}_tauBeta{args.tau_beta}_tauConfig{args.tau_config}_tauSigma{args.tau_sigma}_tauSD{args.tau_sd}'
    f'_IS{args.n_trial_samples}_iters{args.num_epochs}_BatchSize{args.batch_size}_lr{args.lr}_patience{args.scheduler_patience}'
    f'_factor{args.scheduler_factor}_threshold{args.scheduler_threshold}_notes-{args.notes}')
output_dir = os.path.join(output_dir, args.folder_name, 'Run_0')
os.makedirs(output_dir, exist_ok=True)
plot_outputs(model, factor_access_train.permute(2, 0, 1), unique_regions, output_dir, 'Train', -2)
# Initialize the model
if init == 'True':
    model.init_ground_truth(beta=data.beta[:, 1:],
                            alpha=inv_softplus_torch(data.alpha),
                            config_peak_offsets=data.config_peak_offsets,
                            trial_peak_offset_proposal_means=trial_offsets_train.squeeze(),
                            trial_peak_offset_covar_ltri=data.trial_peak_offset_covar_ltri,
                            theta=data.theta,
                            pi=F.softmax(data.pi.reshape(args.A, -1), dim=1).flatten(),
                            trial_peak_offset_proposal_sds=sd_init * torch.ones(2 * data.beta.shape[0],
                                                                                dtype=torch.float64,
                                                                                device=model.device))
elif init == 'Rand':
    model.init_random()
elif init == 'Zero':
    model.init_zero()
elif 'Data' in init:
    model.init_from_data(Y=Y_train, factor_access=factor_access_train, sd_init=sd_init, init=the_rest)
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
create_relevant_files(output_dir, output_str, ground_truth=True)
plot_outputs(model, factor_access_train.permute(2, 0, 1), unique_regions, output_dir, 'Train', -1)

# Instantiate the dataset and dataloader
if args.cuda:
    Y_train, factor_access_train = load_tensors((Y_train, factor_access_train), to_cuda=args.cuda)
dataset = CustomDataset(Y_train, factor_access_train)
if args.batch_size == 'All':
    args.batch_size = Y_train.shape[0]
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print(f'folder_name: {args.folder_name}\n\n')
print(output_str)


# torch.autograd.set_detect_anomaly(True)

def train_gradient(batch_ct):
    for Y, access in dataloader:
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
    total_time = 0
    start_time = time.time()
    batch_ct = 0
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.cuda:
            model.cuda()
            Y_train, factor_access_train = load_tensors((Y_train, factor_access_train), to_cuda=args.cuda)
        model.train()
        batch_ct = train_gradient(batch_ct)
        if epoch % args.eval_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                penalty_term = model.compute_penalty_terms(args.tau_beta, args.tau_config, args.tau_sigma, args.tau_sd)
                likelihood_term_train = model.forward(Y_train, factor_access_train)
                model_factor_assignment_train, model_neuron_gains_train = model.infer_latent_variables()
                likelihood_term_test = model.forward(Y_test, factor_access_test)
                model_factor_assignment_test, model_neuron_gains_test = model.infer_latent_variables()

                losses_train.append(((1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_train +
                                     (1 / (args.n_trials * args.n_configs)) * penalty_term).item())
                log_likelihoods_train.append(
                    (1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_train.item())
                losses_test.append(((1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_test +
                                    (1 / (args.n_trials * args.n_configs)) * penalty_term).item())
                log_likelihoods_test.append(
                    (1 / (args.K * args.n_trials * args.n_configs)) * likelihood_term_test.item())
                epoch_train.append(epoch)

                scheduler.step(log_likelihoods_test[-1])

                non_zero_model_train = torch.nonzero(model_factor_assignment_train)
                non_zero_model_test = torch.nonzero(model_factor_assignment_test)
                non_zero_data_train = torch.nonzero(factor_assignment_onehot_train)
                non_zero_data_test = torch.nonzero(factor_assignment_onehot_test)

                beta_model = model.beta[non_zero_model_train[:, 2]]
                beta_data = data.beta[non_zero_data_train[:, 2]]
                beta_mses.append(F.mse_loss(beta_model, beta_data).item())
                alpha_model = F.softplus(model.alpha[non_zero_model_train[:, 2]])
                alpha_data = data.alpha[non_zero_data_train[:, 2]]
                alpha_mses.append(F.mse_loss(alpha_model, alpha_data).item())
                theta_model = model.theta_value()[non_zero_model_train[:, 2]]
                theta_data = data.theta[non_zero_data_train[:, 2]]
                theta_mses.append(F.mse_loss(theta_model, theta_data).item())
                pi_model = model.pi_value(factor_access_train.permute(2, 0, 1))[non_zero_model_train[:, 2]]
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
                ltriLkhd_train.append(
                    (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_train,
                                                                                        model.ltri_matix()).sum().item())
                ltriLkhd_test.append(
                    (1 / (args.n_trials * args.n_configs)) * model.Sigma_log_likelihood(trial_offsets_test,
                                                                                        model.ltri_matix()).sum().item())

        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
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
            Y_train, factor_access_train = load_tensors((Y_train, factor_access_train))
            with torch.no_grad():
                pi = model.pi_value(factor_access_train.permute(2, 0, 1)).numpy().round(3)
                alpha = F.softplus(model.alpha).numpy().round(3)
                theta = model.theta_value().numpy().round(3)
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
            plot_outputs(model, factor_access_train.permute(2, 0, 1), unique_regions, output_dir, 'Train', epoch)
            is_empty = epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            write_losses(beta_mses, 'Test', 'beta_MSE', output_dir, is_empty)
            write_losses(epoch_train, 'Train', 'Epoch', output_dir, is_empty)
            write_losses(alpha_mses, 'Test', 'alpha_MSE', output_dir, is_empty)
            write_losses(theta_mses, 'Test', 'theta_MSE', output_dir, is_empty)
            write_losses(pi_mses, 'Test', 'pi_MSE', output_dir, is_empty)
            write_losses(config_mses, 'Test', 'configoffset_MSE', output_dir, is_empty)
            write_losses(ltri_mses, 'Test', 'ltri_MSE', output_dir, is_empty)
            write_losses(Sigma_mses, 'Test', 'Sigma_MSE', output_dir, is_empty)
            write_losses(proposal_means_mses, 'Test', 'proposal_means_MSE', output_dir, is_empty)
            write_losses(ltriLkhd_train, 'Train', 'ltriLkhd', output_dir, is_empty)
            write_losses(ltriLkhd_test, 'Test', 'ltriLkhd', output_dir, is_empty)
            write_losses(log_likelihoods_batch, 'Batch', 'Likelihood', output_dir, is_empty)
            write_losses(gains_train, 'Train', 'gains_MSE', output_dir, is_empty)
            write_losses(gains_test, 'Test', 'gains_MSE', output_dir, is_empty)
            write_losses(losses_batch, 'Batch', 'Loss', output_dir, is_empty)
            write_losses(epoch_batch, 'Batch', 'Epoch', output_dir, is_empty)
            plot_losses(true_ELBO_train, output_dir, 'Train', 'Likelihood', 10)
            plot_losses(None, output_dir, 'Train', 'Loss', 10)
            plot_losses(true_ELBO_test, output_dir, 'Test', 'Likelihood', 10)
            plot_losses(None, output_dir, 'Test', 'Loss', 10)
            plot_losses(None, output_dir, 'Test', 'beta_MSE')
            plot_losses(None, output_dir, 'Test', 'alpha_MSE')
            plot_losses(None, output_dir, 'Test', 'theta_MSE')
            plot_losses(None, output_dir, 'Test', 'pi_MSE')
            plot_losses(None, output_dir, 'Test', 'configoffset_MSE')
            plot_losses(None, output_dir, 'Test', 'ltri_MSE')
            plot_losses(None, output_dir, 'Test', 'Sigma_MSE')
            plot_losses(None, output_dir, 'Test', 'proposal_means_MSE')
            plot_losses(true_offset_penalty_train, output_dir, 'Train', 'ltriLkhd', 10)
            plot_losses(true_offset_penalty_test, output_dir, 'Test', 'ltriLkhd', 10)
            plot_losses(true_ELBO_train, output_dir, 'Batch', 'Likelihood', 20)
            plot_losses(None, output_dir, 'Batch', 'Loss', 20)
            plot_losses(None, output_dir, 'Train', 'gains_MSE')
            plot_losses(None, output_dir, 'Test', 'gains_MSE')
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
            print(output_str)
            start_time = time.time()
            if scheduler._last_lr[0] < 1e-5:
                print('Learning rate is too low. Stopping training.')
                break
