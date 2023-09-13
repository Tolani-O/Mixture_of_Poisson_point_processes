import numpy as np
import matplotlib.pyplot as plt

from src.psplines_gradient_method.generate_bsplines import generate_bspline_matrix


def create_precision_matrix(P):
    Omega = np.zeros((P, P))
    # fill the main diagonal with 2s
    np.fill_diagonal(Omega, 2)
    # fill the subdiagonal and superdiagonal with -1s
    np.fill_diagonal(Omega[1:], -1)
    np.fill_diagonal(Omega[:, 1:], -1)
    # set the last element to 1
    Omega[-1, -1] = 1
    return Omega

def create_first_diff_matrix(P):
    D = np.zeros((P-1, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 1:], 1)
    return D


def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # set the last element to 1
    D[-1, -1] = 1
    return D


def create_masking_matrix(N, M):
    mat = np.zeros((N, N * M), dtype=int)  # Start with a kxkm matrix of zeros
    rows = np.repeat(np.arange(N), M)
    cols = np.arange(N * M).reshape(N, M).ravel()
    mat[rows, cols] = 1
    return mat


def compute_lambda(B_psi, d, G_star, beta):
    K = len(d)
    J = np.ones((K, B_psi.shape[1]))
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G_star @ np.kron(np.eye(K), beta) @ B_psi
    lambda_ = np.exp(diagdJ_plus_GBetaB)
    return lambda_


def compute_numerical_grad(Y, B, d, G_star, beta, psi, Omega_beta, Omega_psi,
                           tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt, obj_func, eps=1e-6):
    """Compute numerical gradient of func w.r.t G"""
    G_star_grad = np.zeros_like(G_star)
    beta_grad = np.zeros_like(beta)
    d_grad = np.zeros_like(d)
    psi_grad = np.zeros_like(psi)
    result = obj_func(Y, B, d, G_star, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
    lk1 = result["loss"]
    lk1_G = result["log_likelihood"] + result["beta_penalty"] + result["d_penalty"]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            orig = G[i, j]
            G[i, j] = orig + eps
            result = obj_func(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
            lk2 = result["log_likelihood"] + result["beta_penalty"] + result["d_penalty"]
            G[i, j] = orig
            G_grad[i, j] = (lk2 - lk1_G) / eps

    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            orig = beta[i, j]
            beta[i, j] = orig + eps
            result = obj_func(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
            lk2 = result["loss"]
            beta[i, j] = orig
            beta_grad[i, j] = (lk2 - lk1) / eps

    for i in range(d.shape[0]):
        orig = d[i]
        d[i] = orig + eps
        result = obj_func(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
        lk2 = result["loss"]
        d[i] = orig
        d_grad[i] = (lk2 - lk1) / eps

    return d_grad, G_grad, beta_grad


def plot_spikes(spikes, x_offset=0):
    # Group entries by unique values of s[0]
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset)/1000
        grouped_s.append((i, values))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.show()


def plot_binned(binned):
    # plot binned spikes
    _, ax = plt.subplots()
    ax.imshow(binned)
    ax.invert_yaxis()
    plt.show()


def plot_bsplines(B, time):
    # plot bsplines
    start = 190
    num_basis = 10
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.show()
