import torch
import numpy as np
from torch import distributions

import matplotlib.pyplot as plt


def stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


def plot_prior_samples(model, num_of_samples=1000, save_path=None, observations=None, test_observations=None):
    with torch.no_grad():
        zs = model.sample_p(batch_size=num_of_samples).squeeze()
        t, zs = model.time_grid.cpu().numpy(), zs.cpu().numpy()
        m = zs.mean(axis=1)
        std = zs.std(axis=1)

        plt.subplots(1, 1, figsize=(15, 5))
        for z in zs.T:
            plt.plot(t, z, color='gray', alpha=0.1)

        plt.plot(t, m, color="tab:blue")
        plt.plot(t, m + 2 * std, color="tab:blue")
        plt.plot(t, m - 2 * std, color="tab:blue")

        if observations is not None:
            plt.plot(observations[0], observations[1], "x", color="black")

        if test_observations is not None:
            plt.plot(test_observations[0], test_observations[1], "x", color="tab:red")

        plt.xlabel('$t$')
        plt.ylabel('$Y_t$')
        plt.tight_layout()
        plt.xlim([t[0], t[-1]])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


def plot_posterior_samples(model, num_of_samples=1000, save_path=None, observations=None, test_observations=None):
    with torch.no_grad():
        zs = model.sample_q(batch_size=num_of_samples).squeeze()
        t, zs = model.time_grid.cpu().numpy(), zs.cpu().numpy()

        m = zs.mean(axis=1)
        std = zs.std(axis=1)

        plt.subplots(1, 1, figsize=(15, 5))
        for z in zs.T:
            plt.plot(t, z, color='gray', alpha=0.1)

        if observations is not None:
            plt.plot(observations[0], observations[1], "x", color="black")

        if test_observations is not None:
            plt.plot(test_observations[0], test_observations[1], "x", color="tab:red")

        plt.plot(t, m, color="tab:blue")
        plt.plot(t, m + 2 * std, color="tab:blue")
        plt.plot(t, m - 2 * std, color="tab:blue")

        plt.xlabel('$t$')
        plt.ylabel('$Y_t$')
        plt.tight_layout()
        plt.xlim([t[0], t[-1]])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


def load_data(data_path: str):
    """
    Load data for exp data from a npz file.
    """

    data = np.load(data_path)

    Q = data["Q"]
    x0 = data["x0"]
    noise_stddev = data["sigma"].reshape((1, 1))
    latent_process = data["latent_process"]
    observations = (data["observation_grid"], data["observations"])
    test_obs = (data["test_grid"], data["test_observations"])
    time_grid = data["time_grid"]

    return Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs


def get_optimal_gaussian(model, num_of_samples=1000):
    with torch.no_grad():
        zs = model.sample_q(batch_size=num_of_samples).squeeze()

    m = zs.mean(dim=1)
    std = zs.std(dim=1)

    return distributions.Normal(loc=m, scale=std)


def calculate_nlpd(gaussian_posterior: distributions.Normal, test_observations_y, test_idx):
    if test_observations_y.shape[0] == 0:
        return 0

    m = gaussian_posterior.mean
    s = gaussian_posterior.scale

    m_test = m[test_idx]
    s_test = s[test_idx]

    nlpd = -1 * distributions.Normal(loc=m_test, scale=s_test).log_prob(test_observations_y)
    return nlpd.mean().item()
