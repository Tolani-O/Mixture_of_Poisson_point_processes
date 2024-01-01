import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.TCN.tcn import CustomDataset
from src.simulate_data import DataAnalyzer
import numpy as np
from src.TCN.general_functions import create_second_diff_matrix, plot_spikes, \
    plot_intensity_and_latents, plot_latent_coupling, create_relevant_files, \
    get_parser
from src.TCN.main_learn_initial_outputs import init_initialize_output_models, learn_initial_outputs
from src.TCN.main_learn_initial_maps import init_initialize_map_models, learn_initial_maps
from src.TCN.main_finetune_maps import init_finetune_models, finetune_maps
from scipy.interpolate import BSpline


args = get_parser().parse_args()

# args.K = 200
# args.R = 50
# args.stage = 'finetune'
# args.load = 0
# folder_name = 'paramSeed1029427283_dataSeed2219104129_L3_K200_R50_int.mltply25_int.add1_tauBeta3000_tauS10000_iters25000_notes-empty'
# args.num_epochs = 25000
# args.lr = 1e-4
# if args.stage == 'initialize_output':
#     args.lr = 1e-2
# args.tau_beta = 3000
# args.tau_s = 10000
# args.tau_f = 800
# loss_load_epoch = args.num_epochs - 1
# model_load_epoch = 4300 #args.num_epochs - 1


if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
args.data_seed = np.random.randint(0, 2 ** 32 - 1)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.data_seed)
data_train = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply,
                                 intensity_bias=args.intensity_bias, max_offset=0)
true_likelihood_train = data_train.likelihood()
print(f"True likelihood Training: {true_likelihood_train}")
binned, stim_time = data_train.sample_data()
dims = binned.shape
X_train_array = binned.reshape(dims[0], args.R, int(dims[1] / args.R)).transpose(0, 2, 1)
X_train = [torch.Tensor(X_train_array[i].astype(np.float64)) for i in range(dims[0])]
degree = 3
knots = np.concatenate([np.repeat(stim_time[0], degree), stim_time, np.repeat(stim_time[-1], degree)])
Bspline_matrix = torch.tensor(BSpline.design_matrix(stim_time, knots, degree).transpose().toarray()).float()
state_size = (args.L, Bspline_matrix.shape[0])
Delta2 = create_second_diff_matrix(stim_time.shape[0])
Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2).float()

# create some validation data
data_test = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply,
                                     intensity_bias=args.intensity_bias, max_offset=0)
true_likelihood_test = data_test.likelihood()
print(f"True likelihood Test: {true_likelihood_test}")
binned_test, stim_time = data_test.sample_data()
dims = binned_test.shape
X_test_array = binned_test.reshape(dims[0], args.R, int(dims[1] / args.R)).transpose(0, 2, 1)
X_test = [torch.Tensor(X_test_array[i].astype(np.float64)) for i in range(dims[0])]

start_epoch = 0
model = None
model_optimizer = None
loss_function = None
loss_optimizer = None
output_dir = os.path.join(os.getcwd(), 'outputs')
if args.stage == 'initialize_output':
    model, loss_function, start_epoch, folder_name = init_initialize_output_models(globals())
else:
    # Instantiate the dataset and dataloader
    dataset = CustomDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
if args.stage == 'initialize_map':
    model, loss_function, start_epoch, folder_name = init_initialize_map_models(globals())
if args.stage == 'finetune' or args.stage == 'endtoend':
    model, loss_function, start_epoch, folder_name = init_finetune_models(globals())

if not args.load:
    args.train = 1
if not args.train:
    sys.exit()

print(f'folder_name: {folder_name}')
output_dir = os.path.join(output_dir, folder_name,
                          f'{args.stage}_tauBeta{args.tau_beta}_tauS{args.tau_s}_batchSize{args.batch_size}'
                          f'_lr{args.lr}_iters{args.num_epochs}_notes-{args.notes}')

if not args.load:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        plot_spikes(binned, output_dir)
        plot_intensity_and_latents(data_train.time, data_train.latent_factors, data_train.intensity, output_dir)
        plot_latent_coupling(data_train.latent_coupling, output_dir)
        create_relevant_files(output_dir, args, args.data_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        model.cuda()
        loss_function.cuda()

loss_optimizer = getattr(optim, args.optim)(loss_function.parameters(), lr=args.lr)
if args.stage != 'initialize_output':
    model_optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

# # Visualize networks
# cluster_attn = F.softmax(torch.randn(1, state_size[0]), dim=-1)
# firing_attn = F.softplus(torch.randn(1, state_size[0]))
# latent_coeffs = F.softplus(torch.randn(state_size))
# torch.onnx.export(model, X_train[0].unsqueeze(0), 'runs/model/model.onnx', input_names=["features"], output_names=["outputs"])
# torch.onnx.export(loss_function, (X_train[0].unsqueeze(0), latent_coeffs, cluster_attn, firing_attn),
#                   'runs/loss/loss.onnx', input_names=["features"], output_names=["outputs"])
# # Initialize a writer
# # For the model graph
# model_writer = SummaryWriter('runs/model')
# model_writer.add_graph(model, X_train[0].unsqueeze(0))
# model_writer.close()
# # For the loss graph
# loss_writer = SummaryWriter('runs/loss')
# loss_writer.add_graph(loss_function, [X_train[0].unsqueeze(0), latent_coeffs, cluster_attn, firing_attn])
# loss_writer.close()
# print('DONE')

if __name__ == "__main__":

    if args.stage == 'initialize_output':
        learn_initial_outputs(globals())
        sys.exit()

    if args.stage == 'initialize_map':
        learn_initial_maps(globals())
        sys.exit()

    if args.stage == 'finetune' or args.stage == 'endtoend':
        finetune_maps(globals())
        sys.exit()

        # if epoch > args.log_interval and len(log_likelihoods_test) > 5 and log_likelihoods_test[-1] > max(log_likelihoods_test[-6:-1]):
        #     lr /= 10
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # # Plot activations
        # act = model.activations['conv1'].squeeze()
        # plt.figure(figsize=(15, 5))
        # for i in range(act.size(0)):  # Iterate over all features/neurons
        #     plt.plot(act[i].cpu().numpy(), label=f'Feature {i}')
        # plt.title('Feature Activations Over Sequence')
        # plt.xlabel('Sequence Position')
        # plt.ylabel('Activation')
        # plt.legend()
        # plt.show()
        #
        # # Using seaborn for heatmap
        # plt.figure(figsize=(10, 5))
        # sns.heatmap(act.cpu().numpy(), cmap='viridis')
        # plt.title('Activation Heatmap')
        # plt.xlabel('Sequence Position')
        # plt.ylabel('Feature/Neuron')
        # plt.show()
