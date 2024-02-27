import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, distributions

from model import LatentSDE
from docs.diffusion_processes.neuralsde.neural_sde_exp_utils import plot_prior_samples, load_data, plot_posterior_samples, get_optimal_gaussian, calculate_nlpd


def get_h():
    return lambda t, y: 4 * y * (1 - y**2)


def run_neural_sde(data_dir, output_dir, n_iterations, training_samples, optim_lr, load_model_path=""):
    output_dir = os.path.join(output_dir, data_dir.split(os.path.sep)[-1].split(".")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs = load_data(data_dir)

    time_grid = torch.Tensor(time_grid).to(device)
    h = get_h()

    x0_var = 1e-4

    # Model.
    model = LatentSDE(x0_mu=x0.item(), x0_var=x0_var, Q=Q.item(), time_grid=time_grid, h=h).to(device)
    optimizer = optim.Adam(model.parameters(), lr=optim_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)

    if load_model_path != "":
        model.load_state_dict(torch.load(load_model_path, map_location=device)["model"])
        optimizer.load_state_dict(torch.load(load_model_path, map_location=device)["optimizer"])

    plot_prior_samples(model, save_path=os.path.join(output_dir, "prior_samples.png"), observations=observations,
                       test_observations=test_obs)

    # zs for observations
    time_grid_tmp = np.round(np.array(model.time_grid.cpu().numpy(), dtype=observations[0].dtype), decimals=3)
    obs_sites_indices = np.where(np.equal(time_grid_tmp[..., None], observations[0]))[0]

    t = model.time_grid.cpu()
    time_grid_tmp = np.round(np.array(t.numpy(), dtype=test_obs[0].dtype), decimals=3)
    test_idx = np.where(np.equal(time_grid_tmp[..., None], test_obs[0]))[0]

    observations_y = torch.Tensor(observations[1]).to(device)
    likelihood_constructor = distributions.Normal
    test_observations_y = torch.Tensor(test_obs[1].reshape((-1,))).to(device)

    elbo_vals = []
    nlpd_vals = []
    for global_step in range(n_iterations):
        optimizer.zero_grad()
        zs, kl = model(batch_size=training_samples)
        zs = zs.squeeze()

        zs_obs = zs[obs_sites_indices]
        likelihood = likelihood_constructor(loc=zs_obs, scale=noise_stddev.item())
        logpy = likelihood.log_prob(observations_y).sum(dim=0).mean(dim=0)

        loss = -logpy + kl
        loss.backward()
        optimizer.step()
        scheduler.step()

        elbo_vals.append(loss.item())

        posterior_optimal_gaussian = get_optimal_gaussian(model, num_of_samples=1000)
        nlpd = calculate_nlpd(posterior_optimal_gaussian, test_observations_y=test_observations_y, test_idx=test_idx)
        nlpd_vals.append(nlpd)

        if global_step % 10 == 0:
            print(f"{global_step} = ELBO: {elbo_vals[-1]}; NLPD: {nlpd_vals[-1]}")

    plot_posterior_samples(model, save_path=os.path.join(output_dir, "posterior_samples.png"),
                           observations=observations, test_observations=test_obs)

    np.savez(os.path.join(output_dir, "training_statistics.npz"), elbo=elbo_vals, nlpd=nlpd_vals)

    with torch.no_grad():
        samples = model.sample_q(batch_size=1000).squeeze()
    samples = samples.cpu().numpy()
    np.savez(os.path.join(output_dir, "posterior.npz"), samples=samples)

    plt.plot(elbo_vals)
    plt.title("ELBO")
    plt.show()

    plt.plot(nlpd_vals)
    plt.title("NLPD")
    plt.show()

    torch.save(
        {'model': model.state_dict(),
         'optimizer': optimizer.state_dict()
         },
        os.path.join(output_dir, f'model.ckpt')
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--load_model_path", type=str, default="")
    args.add_argument("--output_dir", type=str, default="outputs/")
    args.add_argument("--n_iterations", type=int, default=10000)
    args.add_argument("--training_samples", type=int, default=100)
    args.add_argument("--optim_lr", type=float, default=0.1)

    args = args.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    n_iterations = args.n_iterations
    training_samples = args.training_samples
    optim_lr = args.optim_lr
    load_model_path = args.load_model_path

    run_neural_sde(data_dir, output_dir, n_iterations, training_samples, optim_lr, load_model_path)
