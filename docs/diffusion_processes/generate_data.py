"""
Generate observations from Ornstein-Uhlenbeck process and save in a npz file.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from gpflow.config import default_float
import matplotlib.pyplot as plt

from markovflow.sde.sde_utils import euler_maruyama
from markovflow.sde.sde import OrnsteinUhlenbeckSDE, SDE, DoubleWellSDE, BenesSDE, SineDiffusionSDE, SqrtDiffusionSDE, VanderPolOscillatorSDE
from docs.diffusion_processes.exp_dp_utils import get_k_folds

DTYPE = default_float()


def set_seed(seed: int):
    """Set the seed"""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_observations(sde: SDE, t0: float, t1: float, x0: tf.Tensor, dt: float,
                     noise_std: float, num_observations: int) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    Simulate an OU-SDE using Euler-Maruyama and return observations.
    """
    n_decimals = str(dt)[::-1].find('.')
    time_grid = np.round(tf.range(t0, t1 + dt, dt, dtype=DTYPE), decimals=n_decimals)
    simulated_vals = euler_maruyama(sde, x0=x0, time_grid=time_grid)[0]

    observation_grid = tf.convert_to_tensor(
        np.sort(np.random.choice(time_grid, num_observations, replace=False)).reshape((-1,)), dtype=DTYPE)
    observation_idx = tf.where(tf.equal(time_grid[..., None], observation_grid))[:, 0]
    observations = tf.gather(simulated_vals, observation_idx, axis=0)
    observations = observations + tf.random.normal(observations.shape, stddev=noise_std, dtype=DTYPE)

    # Get test set data
    test_observation_grid = tf.convert_to_tensor(
        np.sort(np.random.choice(time_grid, int(.2 * num_observations), replace=False)).reshape((-1,)), dtype=DTYPE)
    test_observation_idx = tf.where(tf.equal(time_grid[..., None], test_observation_grid))[:, 0]
    test_observations = tf.gather(simulated_vals, test_observation_idx, axis=0)
    test_observations = test_observations + tf.random.normal(test_observations.shape, stddev=noise_std, dtype=DTYPE)

    return simulated_vals, observation_grid, observations, time_grid, test_observation_grid, test_observations


def plot_data(latent_grid, latent_process, observations, test_observations):
    """
    Plot the observations.
    """

    for i in range(latent_process.shape[-1]):
        plt.subplots(1, 1, figsize=(15, 5))
        plt.plot(observations[0], observations[1][:, i], "kx")
        plt.plot(test_observations[0], test_observations[1][:, i], "x", color="red")
        plt.plot(latent_grid, latent_process[:, i], alpha=0.2, color="gray")
        plt.show()


if __name__ == '__main__':
    """
    python generate_data.py -d=0.5 -q=0.8 -t0=0. -t1=10. -x0=0. -dt=0.01 -n=30 --o=data/33.npz  
    
    Number of test data = number of observations * 0.2
    
    """
    parser = argparse.ArgumentParser(description="Generate observations from an SDE.")
    parser.add_argument("-sde", help="SDE to simulate the process", choices=['ou', 'dw', 'benes', "sine",
                                                                             "sqrt", "vanderpol"],
                        required=True)
    parser.add_argument("-d", "--decay", help="Decay for the Ornstein-Uhlenbeck process.", default=0.5, type=float)
    parser.add_argument("-q", "--diffusion", help="Spectral density for the diffusion process.", type=float,
                        default=0.8)
    parser.add_argument("-t0", help="Time t0.", type=float, default=0.0)
    parser.add_argument("-t1", help="Time t1.", type=float, default=1.0)
    parser.add_argument("-x0", help="State at t0.", type=float, default=0.0)
    parser.add_argument("-dt", help="Time step value (dt)", type=float, default=0.01)
    parser.add_argument("-n", "--num_observations", help="Number of Observations", type=int, default=10)
    parser.add_argument("-si", "--sigma", help="Noise std-deviation", type=float, default=0.1)
    parser.add_argument("-o", "--output", help="Output directory path.", type=str, default="")
    parser.add_argument("-s", "--seed", help="Set the seed.", type=int, default=33)
    parser.add_argument("-k", "--kfolds", help="Number of k-folds set to output.", type=int, default=0)
    parser.add_argument("-dim", help="Number of state dimensions.", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)

    q = args.diffusion * np.eye(args.dim, dtype=DTYPE)
    x0 = args.x0 * np.ones((1, args.dim), dtype=DTYPE)

    if args.sde == "ou":
        sde = OrnsteinUhlenbeckSDE(decay=args.decay, q=q)
    elif args.sde == "dw":
        sde = DoubleWellSDE(q=q)
    elif args.sde == "benes":
        sde = BenesSDE(q=q)
    elif args.sde == "sine":
        sde = SineDiffusionSDE(q=q)
    elif args.sde == "sqrt":
        sde = SqrtDiffusionSDE(q=q)
    elif args.sde == "vanderpol":
        sde = VanderPolOscillatorSDE(a=2., tau=5., q=q)
    else:
        raise Exception("SDE is not supported!")

    simulated_vals, obs_grid, obs_vals, time_grid, test_obs_grid, test_obs_vals = get_observations(sde=sde, t0=args.t0,
                                                                                                   t1=args.t1,
                                                                                                   x0=x0,
                                                                                                   num_observations=args.num_observations,
                                                                                                   dt=args.dt,
                                                                                                   noise_std=args.sigma)

    if args.output == "":
        output_path = str(args.seed)
    else:
        output_path = os.path.join(args.output, str(args.seed))

    if args.kfolds > 0:
        print(f"Creating {args.kfolds} k-fold datasets...")
        k_folds_train_data, k_folds_test_data = get_k_folds((obs_grid, obs_vals), args.kfolds)
        os.makedirs(output_path)

        for fold_id, (obs_vals, test_obs_vals) in enumerate(zip(k_folds_train_data, k_folds_test_data)):
            np.savez(os.path.join(output_path, f"{fold_id}"), sde=args.sde, decay=args.decay, Q=q, x0=x0, sigma=args.sigma,
                     latent_process=simulated_vals, observations=obs_vals[1], observation_grid=obs_vals[0],
                     time_grid=time_grid, test_observations=test_obs_vals[1], test_grid=test_obs_vals[0])

            print(f"Number of observations (k_fold = {fold_id}) = {obs_vals[0].shape[0]}")
            print(f"Number of test-observations (k_fold = {fold_id}) = {test_obs_vals[0].shape[0]}")

            plot_data(time_grid, simulated_vals, obs_vals, test_obs_vals)

    else:
        np.savez(output_path, sde=args.sde, decay=args.decay, Q=q, x0=x0, sigma=args.sigma,
                 latent_process=simulated_vals, observations=obs_vals, observation_grid=obs_grid,
                 time_grid=time_grid, test_observations=test_obs_vals, test_grid=test_obs_grid)

        print(f"Number of observations = {obs_grid.shape[0]}")
        print(f"Number of test-observations = {test_obs_grid.shape[0]}")

        plot_data(time_grid, simulated_vals, (obs_grid, obs_vals), (test_obs_grid, test_obs_vals))
