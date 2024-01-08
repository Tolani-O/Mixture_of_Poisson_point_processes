import os
import time
import torch
import torch.nn.functional as F
from src.TCN.general_functions import write_log_and_model, plot_outputs, write_losses, plot_losses, \
    load_model_checkpoint, reset_metric_checkpoint
from src.TCN.tcn import TemporalConvNet


def define_global_vars(global_vars):
    global model, loss_function, model_optimizer, dataloader, data_train, data_test, X_train, X_test, stim_time, args, \
        output_dir, start_epoch, Bspline_matrix, Delta2TDelta2, state_size
    model = global_vars['model']
    loss_function = global_vars['loss_function']
    model_optimizer = global_vars['model_optimizer']
    dataloader = global_vars['dataloader']
    data_train = global_vars['data_train']
    data_test = global_vars['data_test']
    X_train = global_vars['X_train']
    X_test = global_vars['X_test']
    stim_time = global_vars['stim_time']
    args = global_vars['args']
    output_dir = global_vars['output_dir']
    start_epoch = global_vars['start_epoch']
    Bspline_matrix = global_vars['Bspline_matrix']
    Delta2TDelta2 = global_vars['Delta2TDelta2']
    state_size = global_vars['state_size']


def init_initialize_map_models(global_vars):
    define_global_vars(global_vars)
    folder_name = args.init_load_folder
    # We must load a loss checkpoint from 'initialize_output'
    sub_folder_name = args.init_load_subfolder_outputs
    loss_function = load_model_checkpoint('loss', output_dir, folder_name, sub_folder_name, args.init_loss_load_epoch)
    if args.load:
        # This says to load a continue checkpoint for model from 'initialize_map'
        sub_folder_name = args.init_load_subfolder_map
        model = load_model_checkpoint('model', output_dir, folder_name, sub_folder_name, args.init_map_load_epoch)
        start_epoch = args.init_map_load_epoch
        if args.reset_checkpoint:
            metric_files = ['losses_test', 'losses_train']
            reset_metric_checkpoint(output_dir, folder_name, sub_folder_name, metric_files, start_epoch)
            _, cluster_attn_train, firing_attn_train = initialization_evaluation(X_train, [])
            _, cluster_attn_test, firing_attn_test = initialization_evaluation(X_test, [])
            plot_dir = os.path.join(output_dir, folder_name, sub_folder_name)
            latent_coeffs = F.softplus(loss_function.state.detach())
            latent_factors = torch.matmul(latent_coeffs, Bspline_matrix).numpy()
            cluster_attn = cluster_attn_train.detach().numpy()
            firing_attn = firing_attn_train.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, plot_dir, 'Train', start_epoch)
            cluster_attn = cluster_attn_test.detach().numpy()
            firing_attn = firing_attn_test.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, plot_dir, 'Test', start_epoch)
            plot_losses(0, plot_dir, 'Train', 'Loss', 20)
            plot_losses(0, plot_dir, 'Test', 'Loss', 20)
    else:
        if isinstance(args.param_seed, int):
            torch.manual_seed(args.param_seed)
        n_channels = [args.nhid] * args.levels
        input_size = args.R
        output_size = args.nhid
        bin_kernel_size = 2
        model = TemporalConvNet(state_size, input_size, output_size, n_channels, output_size, bin_kernel_size,
                                args.ksize, args.dropout)
        # Attach hooks
        # model.register_hooks()
        model.init_from_states(loss_function.state.clone())
        if isinstance(args.param_seed, str) and args.param_seed.lower() == 'truth':
            loss_function.init_from_factors(torch.tensor(data_train.latent_factors).float())
            model.init_from_factors(torch.tensor(data_train.latent_factors).float(), Bspline_matrix)
        start_epoch = 0
    loss_function.state.requires_grad = False
    loss_function.cluster_attn.requires_grad = False
    loss_function.firing_attn.requires_grad = False
    return model, loss_function, start_epoch, folder_name


def initialization_training_epoch(losses):
    model.train()
    for binned, idx in dataloader:
        if args.cuda: binned = binned.cuda()
        # Zero out the gradients for optimizer
        model_optimizer.zero_grad()
        latent_coeffs, cluster_attn, firing_attn = model(binned)
        loss = loss_function(cluster_attn=cluster_attn, firing_attn=firing_attn, idx=idx, mode='initialize_map')
        losses.append(-loss.item())
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # Backpropagate the loss through both the model and the loss_function
        loss.backward()
        # Update the parameters for both the model and the loss_function
        model_optimizer.step()
    return losses


def initialization_evaluation(data, losses):
    model.eval()
    with torch.no_grad():
        data = torch.stack(data)
        if args.cuda: data = data.cuda()
        latent_coeffs, cluster_attn, firing_attn = model(data)
        loss = loss_function(cluster_attn=cluster_attn, firing_attn=firing_attn, mode='initialize_map')
    losses.append(-loss.item())
    return losses, cluster_attn, firing_attn


def learn_initial_maps(global_vars):
    define_global_vars(global_vars)
    total_time = 0
    losses = []
    losses_train = []
    losses_test = []
    start_time = time.time()  # Record the start time of the epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        losses = initialization_training_epoch(losses)
        losses_train, cluster_attn_train, firing_attn_train = initialization_evaluation(X_train, losses_train)
        losses_test, cluster_attn_test, firing_attn_test = initialization_evaluation(X_test, losses_test)
        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_loss_train = losses_train[-1]
            cur_loss_test = losses_test[-1]
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Loss test: {cur_loss_test:.5f}, lr: {args.lr:.5f}\n")
            write_log_and_model(output_str, output_dir, epoch, model=model)
            latent_coeffs = F.softplus(loss_function.state.detach())
            latent_factors = torch.matmul(latent_coeffs, Bspline_matrix).numpy()
            cluster_attn = cluster_attn_train.detach().numpy()
            firing_attn = firing_attn_train.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir, 'Train', epoch)
            cluster_attn = cluster_attn_test.detach().numpy()
            firing_attn = firing_attn_test.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir, 'Test', epoch)
            is_empty = start_epoch == 0 and epoch == 0
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            write_losses(losses, 'Batch', 'Loss', output_dir, is_empty)
            plot_losses(0, output_dir, 'Train', 'Loss', 20)
            plot_losses(0, output_dir, 'Test', 'Loss', 20)
            plot_losses(0, output_dir, 'Batch', 'Loss', 100)
            losses_train = []
            losses_test = []
            start_time = time.time()  # Record the start time of the epoch
            print(output_str)
