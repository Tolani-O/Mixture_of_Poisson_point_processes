import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SimpleAttentionScores(nn.Module):
    def __init__(self, scale=None):
        super(SimpleAttentionScores, self).__init__()
        self.scale = scale

    def forward(self, query, key):
        padding_size = key.size(1) - query.size(1)
        query_padded = F.pad(query, (0, padding_size))
        # Compute the dot product between query and key
        attention_scores = torch.matmul(query_padded, key.transpose(0, 1))
        # Apply scaling
        if self.scale:
            attention_scores = attention_scores / self.scale

        return attention_scores


class ConvolutionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ConvolutionBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.elu, self.dropout)

    def forward(self, x):
        return self.net(x)


class EmbeddingConvolutionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(EmbeddingConvolutionBlock, self).__init__()
        conv1 = ConvolutionBlock(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        conv2 = ConvolutionBlock(n_outputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        self.convnet = nn.Sequential(conv1, conv2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.convnet(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class NegLogLikelihood(nn.Module):
    def __init__(self, state_size, batch_size, Bspline_matrix, Delta2TDelta2, del_t):
        super(NegLogLikelihood, self).__init__()
        # Initialize any parameters here
        self.smoothness_budget = nn.Parameter(torch.zeros((state_size[0],1)))
        self.Bspline_matrix = Bspline_matrix
        self.Delta2TDelta2 = Delta2TDelta2
        self.del_t = del_t
        # Initialize the TCN Network's transformed parameters
        self.state = nn.Parameter(torch.cat((torch.zeros(1, state_size[1]), torch.randn(state_size[0] - 1, state_size[1])), dim=0))
        self.cluster_attn = nn.Parameter(torch.randn(batch_size, state_size[0]))
        self.firing_attn = nn.Parameter(torch.randn(batch_size, 1))
        # Create loss
        self.mse_loss = nn.MSELoss()

    def init_from_factors(self, latent_factors):
        V_inv = torch.pinverse(self.Bspline_matrix)
        beta = latent_factors @ V_inv
        states = torch.log(torch.exp(beta) - 1)
        self.state = nn.Parameter(states)
        self.smoothness_budget = nn.Parameter(torch.zeros((self.smoothness_budget.shape[0], 1)))

    def likelihood_term(self, spike_trains, latent_factors, cluster_attn, firing_attn):
        # Compute the loss
        weighted_firing = cluster_attn * firing_attn
        intensity_functions = torch.matmul(weighted_firing, latent_factors).unsqueeze(2)
        intensity_functions_repeated = intensity_functions.repeat_interleave(spike_trains.shape[2], dim=2)
        negLogLikelihood = -torch.sum(torch.log(intensity_functions_repeated) * spike_trains - intensity_functions * self.del_t)
        return negLogLikelihood

    def penalty_term(self, latent_factors, tau_beta, tau_s):
        smoothness_budget_constrained = F.softmax(self.smoothness_budget, dim=0)
        beta_s2_penalty = tau_beta * (smoothness_budget_constrained.t() @ torch.sum((latent_factors @ self.Delta2TDelta2) * latent_factors, axis=1)).squeeze()
        smoothness_budget_penalty = tau_s * torch.sum(self.smoothness_budget ** 2)
        penalty = beta_s2_penalty + smoothness_budget_penalty
        return penalty, smoothness_budget_constrained

    def forward(self, spike_trains=None, latent_coeffs=None, cluster_attn=None, firing_attn=None, idx=None, tau_beta=1, tau_s=1, mode=''):
        tau_beta = torch.tensor(tau_beta)
        tau_s = torch.tensor(tau_s)
        if mode=='initialize_output':
            latent_coeffs = F.softplus(self.state)
            latent_factors = torch.matmul(latent_coeffs, self.Bspline_matrix)
            cluster_attn = F.softmax(self.cluster_attn, dim=-1)
            firing_attn = F.softplus(self.firing_attn)
            negLogLikelihood = self.likelihood_term(spike_trains, latent_factors, cluster_attn, firing_attn)
            penalty, smoothness_budget_constrained = self.penalty_term(latent_factors, tau_beta, tau_s)
            loss = negLogLikelihood + penalty
            return loss, negLogLikelihood, latent_factors, cluster_attn, firing_attn, smoothness_budget_constrained
        elif mode=='initialize_map':
            if idx is None:
                idx = torch.arange(cluster_attn.size(0))
            cluster_loss = self.mse_loss(cluster_attn, self.cluster_attn[idx])
            firing_loss = self.mse_loss(firing_attn, self.firing_attn[idx])
            loss = cluster_loss + firing_loss
            return loss
        else:
            latent_factors = torch.matmul(latent_coeffs, self.Bspline_matrix)
            negLogLikelihood = self.likelihood_term(spike_trains, latent_factors, cluster_attn, firing_attn)
            penalty, smoothness_budget_constrained = self.penalty_term(latent_factors, tau_beta, tau_s)
            loss = negLogLikelihood + penalty
            return loss, negLogLikelihood, latent_factors, smoothness_budget_constrained



class TemporalConvNet(nn.Module):
    def __init__(self, state_size, num_inputs, num_bin_outputs, num_embed_channels, num_embed_outputs, bin_kernel_size=2, embed_kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        # Initialize the learnable variable that stores the state
        self.state = nn.Parameter(torch.cat((torch.zeros(1, state_size[1]), torch.randn(state_size[0] - 1, state_size[1])), dim=0))

        # binning convolutions
        conv1 = ConvolutionBlock(num_inputs, num_bin_outputs, bin_kernel_size, 1, 1, (bin_kernel_size - 1), dropout)
        conv2 = ConvolutionBlock(num_bin_outputs, num_bin_outputs, bin_kernel_size, 1, 1, (bin_kernel_size - 1), dropout)
        self.binningConv = nn.Sequential(conv1, conv2)

        # latent state convolutions
        conv1 = ConvolutionBlock(1, num_bin_outputs, bin_kernel_size, 1, 1, (bin_kernel_size - 1), dropout)
        conv2 = ConvolutionBlock(num_bin_outputs, num_bin_outputs, bin_kernel_size, 1, 1, (bin_kernel_size - 1), dropout)
        self.stateConv = nn.Sequential(conv1, conv2)

        # Embedding Convolutions
        cluster_layers = []
        firing_layers = []
        num_levels = len(num_embed_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            channels = num_embed_channels[i]
            cluster_layers += [EmbeddingConvolutionBlock(channels, channels, embed_kernel_size, stride=1,
                                                dilation=dilation_size, padding=(embed_kernel_size - 1) * dilation_size,
                                                dropout=dropout)]
            firing_layers += [EmbeddingConvolutionBlock(channels, channels, embed_kernel_size, stride=1,
                                                dilation=dilation_size, padding=(embed_kernel_size - 1) * dilation_size,
                                                dropout=dropout)]
        self.ClusterEmbeddingConv = nn.Sequential(*cluster_layers)
        linear_layers = [nn.Linear(num_embed_channels[-1], num_embed_outputs), nn.Linear(num_embed_channels[-1], 1)]
        self.ClusterEmbeddingLinr = nn.Sequential(*linear_layers)
        self.FiringEmbeddingConv = nn.Sequential(*firing_layers)
        linear_layers = [nn.Linear(num_embed_channels[-1], num_embed_outputs), nn.Linear(num_embed_channels[-1], 1)]
        self.FiringEmbeddingLinr = nn.Sequential(*linear_layers)

        # Attention layer
        self.AttnScores = SimpleAttentionScores()
        # Post attention linear layer
        self.PostAttnFiringLinr = nn.Linear(state_size[0], 1)

    # Will need to train the network on the ground truth to initialize
    # The neuron couplings and firing rates at the ground truth
    def init_from_factors(self, latent_factors, Bspline_matrix):
        V_inv = torch.pinverse(Bspline_matrix)
        beta = latent_factors @ V_inv
        states = torch.log(torch.exp(beta) - 1)
        self.init_from_states(states)

    def init_from_states(self, latent_states):
        self.state = nn.Parameter(latent_states)

    def register_hooks(self):
        def save_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        self.activations = {}  # Dictionary to store activations
        # Attach hooks to self.network
        for name, layer in self.network.named_modules():
            layer.register_forward_hook(save_activation(name))
        # Add hooks for other layers if needed

    def forward(self, x):
        state_output = self.stateConv(self.state.unsqueeze(1))
        bin_output = self.binningConv(x.transpose(1, 2))
        bin_cluster_output = self.ClusterEmbeddingConv(bin_output).transpose(1, 2)
        bin_cluster_output = self.ClusterEmbeddingLinr(bin_cluster_output)
        bin_firing_output = self.FiringEmbeddingConv(bin_output).transpose(1, 2)
        bin_firing_output = self.FiringEmbeddingLinr(bin_firing_output)
        state_cluster_output = self.ClusterEmbeddingConv(state_output).transpose(1, 2)
        state_cluster_output = self.ClusterEmbeddingLinr(state_cluster_output)
        state_firing_output = self.FiringEmbeddingConv(state_output).transpose(1, 2)
        state_firing_output = self.FiringEmbeddingLinr(state_firing_output)
        cluster_attn = self.AttnScores(bin_cluster_output.squeeze(-1), state_cluster_output.squeeze(-1))
        cluster_attn = F.softmax(cluster_attn, dim=-1)
        firing_attn = self.AttnScores(bin_firing_output.squeeze(-1), state_firing_output.squeeze(-1))
        firing_attn = self.PostAttnFiringLinr(firing_attn)
        firing_attn = F.softplus(firing_attn)
        latent_coeffs = F.softplus(self.state)
        return latent_coeffs, cluster_attn, firing_attn


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx
