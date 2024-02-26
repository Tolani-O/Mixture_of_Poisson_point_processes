import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Gamma, Dirichlet, Poisson

class VariationalModel(nn.Module):
    def __init__(self, N, L):
        super(VariationalModel, self).__init__()
        self.N = N  # Number of data points
        self.L = L  # Number of latent variable components

        # Initialize variational parameters for Gamma and Dirichlet distributions
        self.var_theta = nn.Parameter(torch.rand(L))
        self.var_beta = nn.Parameter(torch.rand(L))
        self.var_alpha = nn.Parameter(torch.rand(N, L))

    def sample_latent_variables(self):
        # Sample from the variational distributions
        q_gamma = Gamma(self.var_theta, self.var_beta)
        q_pi = Dirichlet(self.var_alpha)
        return q_gamma.rsample(), q_pi.rsample()

    def compute_elbo(self, x_data, alpha, theta, beta):
        # Sample from the variational distributions
        q_gamma, q_pi = self.sample_latent_variables()

        # Compute the log likelihood
        likelihood = Poisson(q_gamma * q_pi).log_prob(x_data).sum()

        # Compute the KL divergence
        kl_divergence = (q_gamma * (torch.log(q_gamma) - torch.log(theta))).sum() + \
                        (q_pi * (torch.log(q_pi) - torch.log(alpha))).sum()

        # Compute the ELBO as the expectation of the log likelihood minus the KL divergence
        elbo = likelihood - kl_divergence

        return elbo

    def forward(self, x_data, alpha, theta, beta):
        return self.compute_elbo(x_data, alpha, theta, beta)


N = 100  # Number of data points
L = 10   # Number of latent variable components

# Initialize model parameters
alpha = torch.rand(L)
theta = torch.rand(L)
beta = torch.rand(L)

# Initialize the variational model
model = VariationalModel(N, L)

optimizer = Adam(model.parameters(), lr=0.01)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Sample latent variables
    gamma_sample, pi_sample = model.sample_latent_variables()

    # Compute ELBO
    elbo = model(x_data, alpha, theta, beta)

    # Maximize ELBO (minimize negative ELBO)
    loss = -elbo
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item()}")
