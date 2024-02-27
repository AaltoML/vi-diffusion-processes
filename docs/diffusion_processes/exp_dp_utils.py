import logging
from typing import Tuple
import shutil
import os
from distutils.dir_util import copy_tree
import hydra
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.sde.sde_utils import gaussian_log_predictive_density


def plot_posterior(m, S, observation_grid, observation_val, time_grid, latent_process, latent_process_grid=None,
                   output_path=None, test_observations: Tuple[tf.Tensor, tf.Tensor] = None, model_legend: str = None):
    """
    Plot the posterior.
    """
    tf.debugging.assert_shapes(
        [
            [m, ("N", "D")],
            [S, ("N", "D", "D")]
        ]
    )

    if latent_process_grid is None:
        latent_process_grid = time_grid

    D = m.shape[-1]

    fig, axs = plt.subplots(D, 1, figsize=(30, 10))

    if D == 1:
        axs.plot(observation_grid, observation_val, "x", color="black", label="observations")
        if test_observations:
            axs.plot(test_observations[0], test_observations[1], "x", color="red", label="test-observations")
        if latent_process is not None:
            axs.plot(latent_process_grid, latent_process, alpha=0.3, color="black")

        S_i = tf.linalg.diag_part(S)
        axs.plot(time_grid, m, color="tab:blue", label=model_legend)
        axs.plot(time_grid, m + 2 * tf.sqrt(S_i), color="tab:blue")
        axs.plot(time_grid, m - 2 * tf.sqrt(S_i), color="tab:blue")

        axs.set_xlabel("Time (t)")
        axs.set_xlim([time_grid[0], time_grid[-1]])
    else:
        for i in range(D):
            axs[i].plot(observation_grid, observation_val[:, i], "x", color="black", label="observations")
            if test_observations:
                axs[i].plot(test_observations[0], test_observations[1][:, i], "x", color="red", label="test-observations")
            if latent_process is not None:
                axs[i].plot(latent_process_grid, latent_process[:, i], alpha=0.3, color="black")

            S_i = tf.linalg.diag_part(S)[:, i]
            axs[i].plot(time_grid, m[:, i], color="tab:blue", label=model_legend)
            axs[i].plot(time_grid, m[:, i] + 2 * tf.sqrt(S_i), color="tab:blue")
            axs[i].plot(time_grid, m[:, i] - 2 * tf.sqrt(S_i), color="tab:blue")

            axs[i].set_xlabel("Time (t)")
            axs[i].set_xlim([time_grid[0], time_grid[-1]])

    plt.title("Posterior")
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path)

    if wandb.run is not None:
        wandb.log({"Posterior": wandb.Image(plt)})

    plt.show()


def plot_params_of_vi_markov(vi_model: VariationalMarkovGP):
    """
    Plot the params of the Variational Markov GP model.
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    axs[0][0].plot(vi_model.A.numpy().reshape((-1, 1)))
    axs[0][0].set_title("A")

    axs[0][1].plot(vi_model.b.numpy().reshape((-1, 1)))
    axs[0][1].set_title("b")

    axs[1][0].plot(vi_model.lambda_lagrange.numpy().reshape((-1, 1)))
    axs[1][0].set_title("lambda")

    axs[1][1].plot(vi_model.psi_lagrange.numpy().reshape((-1, 1)))
    axs[1][1].set_title("psi")

    plt.show()


def get_hydra_output_dir():
    """Return the current output directory path generated by hydra"""
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg['runtime']['output_dir']


def load_exp_data(data_path: str):
    """
    Load data for exp data from a npz file.
    """

    data = np.load(data_path)

    # decay = data["decay"]
    Q = data["Q"]
    x0 = data["x0"]
    noise_stddev = data["sigma"].reshape((1, 1))
    latent_process = data["latent_process"]
    observations = (tf.convert_to_tensor(data["observation_grid"]), tf.convert_to_tensor(data["observations"]))
    test_obs = (tf.convert_to_tensor(data["test_grid"]), tf.convert_to_tensor(data["test_observations"]))
    time_grid = tf.convert_to_tensor(data["time_grid"])

    return Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs


def load_apple_stock_exp_data(data_path: str):
    """
    Load Apple stock data for exp data from a npz file.
    """

    data = np.load(data_path)
    train_data = data["train_data"]
    test_data = data["test_data"]

    Q = 0.1
    x0 = train_data[0][1]
    noise_stddev = np.ones_like(x0).reshape((1, 1)) * 0.5
    observations = (tf.convert_to_tensor(train_data[:, 2]), tf.convert_to_tensor(train_data[:, 1][..., None]))
    test_obs = (tf.convert_to_tensor(test_data[:, 2]), tf.convert_to_tensor(test_data[:, 1][..., None]))
    time_grid = np.linspace(0, max(train_data[-1, 2], test_data[-1, 2]), train_data.shape[0] + test_data.shape[0])
    n_decimals = 2
    time_grid = tf.convert_to_tensor(np.round(time_grid, n_decimals))
    return Q, x0, noise_stddev, None, observations, time_grid, test_obs


def load_gps_exp_data(data_path: str, train_dim: int = 0):
    """
    Load GPS data for exp data from a npz file.

    train_dim=0 is x dim and train_dim=1 is y dim.

    """

    data = np.load(data_path)

    train_data_t = data["train_data_t"]
    test_data_t = data["test_data_t"]
    if train_dim == 0:
        train_data = (train_data_t, data["train_data_x"])
        test_data = (test_data_t, data["test_data_x"])
    else:
        train_data = (train_data_t, data["train_data_y"])
        test_data = (test_data_t, data["test_data_y"])

    Q = 0.1
    x0 = train_data[1][0]
    noise_stddev = np.ones_like(x0).reshape((1, 1)) * 0.1

    observations = (tf.convert_to_tensor(train_data[0].reshape((-1,))), tf.convert_to_tensor(train_data[1]))
    test_obs = (tf.convert_to_tensor(test_data[0].reshape((-1,))), tf.convert_to_tensor(test_data[1]))
    time_grid = np.concatenate([train_data[0], test_data[0]], axis=0)
    time_grid = tf.convert_to_tensor(np.sort(time_grid, axis=0).reshape((-1, )))
    return Q, x0, noise_stddev, None, observations, time_grid, test_obs


def modify_time_grid(time_grid: tf.Tensor, dt: float) -> tf.Tensor:
    """
    Modify the time-grid considering the new time-step, dt.
    """

    t0 = time_grid[0]
    t1 = time_grid[-1]
    time_grid = tf.range(t0, t1 + dt, dt, dtype=time_grid.dtype)
    n_decimals = str(dt)[::-1].find('.')
    return tf.convert_to_tensor(np.round(time_grid, n_decimals))


def calculate_nlpd(m: tf.Tensor, S: tf.Tensor, time_grid: tf.Tensor, test_data: [tf.Tensor, tf.Tensor]) -> float:
    """
    Calculate negative log predictive density.
    """

    tf.debugging.assert_shapes(
        [
            (m, ("N", "D")),
            (S, ("N", "D", "D"))
        ]
    )

    test_obs_indices = tf.where(tf.equal(time_grid[..., None], test_data[0]))[:, 0]
    m_test = tf.gather(m, test_obs_indices)
    S_test = tf.gather(S, test_obs_indices)
    nlpd = -1 * tf.reduce_mean(gaussian_log_predictive_density(m_test, tf.linalg.cholesky(S_test),
                                                               test_data[1]))
    return nlpd.numpy().item()


def calculate_rmse(m: tf.Tensor, time_grid: tf.Tensor, test_data: [tf.Tensor, tf.Tensor]) -> float:
    """
    Calculate RMSE
    """

    tf.debugging.assert_shapes(
        [
            (m, ("N", "D")),
        ]
    )

    test_obs_indices = tf.where(tf.equal(time_grid[..., None], test_data[0]))[:, 0]
    m_test = tf.gather(m, test_obs_indices)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(m_test - test_data[1])))

    return rmse.numpy().item()


def plot_line(vals: list, output_path=None, title: str = ""):
    """
    Plot the list values and save the plot if output_path is provided using the title.
    """
    plt.clf()
    plt.plot(vals)
    plt.xlabel("Iterations")
    plt.title(title)

    if output_path:
        plt.savefig(output_path)

    plt.show()


def plot_all_posterior(cvi_m: tf.Tensor, cvi_S: tf.Tensor, gpr_m: tf.Tensor, gpr_S: tf.Tensor,
                       vi_m: tf.Tensor, vi_S: tf.Tensor, observation_grid: tf.Tensor, observation_val: tf.Tensor,
                       time_grid: tf.Tensor, latent_process: tf.Tensor):
    """
    Plot all the model's posterior.
    """
    tf.debugging.assert_shapes(
        [
            [cvi_m, ("N", "D")],
            [cvi_S, ("N", "D")],
            [gpr_m, ("N", "D")],
            [gpr_S, ("N", "D")],
            [vi_m, ("N", "D")],
            [vi_S, ("N", "D")]
        ]
    )

    plt.subplots(1, 1, figsize=(15, 5))

    plt.plot(observation_grid, observation_val, "x", color="red", label="observations")
    plt.plot(time_grid, latent_process, alpha=0.3, color="black")

    plt.plot(time_grid, cvi_m, color="tab:blue", label="Proposed")
    plt.plot(time_grid, cvi_m + 2 * tf.sqrt(cvi_S), color="tab:blue")
    plt.plot(time_grid, cvi_m - 2 * tf.sqrt(cvi_S), color="tab:blue")

    plt.plot(time_grid, gpr_m, color="tab:red", label="GPR")
    plt.plot(time_grid, gpr_m + 2 * tf.sqrt(gpr_S), color="tab:red")
    plt.plot(time_grid, gpr_m - 2 * tf.sqrt(gpr_S), color="tab:red")

    plt.plot(time_grid, vi_m, color="tab:green", label="Archambeau et al.")
    plt.plot(time_grid, vi_m + 2 * tf.sqrt(vi_S), color="tab:green")
    plt.plot(time_grid, vi_m - 2 * tf.sqrt(vi_S), color="tab:green")

    plt.xlim([time_grid[0], time_grid[-1]])
    plt.legend()

    return plt


def setup_wandb(cfg):
    """
    Set up wandb if username is passed.
    """
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(project="VI-DP", entity=cfg.wandb.username, config=wandb_cfg, tags=cfg.wandb.tags)
    return run


def get_cross_validation_sets(data: Tuple[tf.Tensor, tf.Tensor], k_folds: int) -> [Tuple[tf.Tensor, tf.Tensor],
                                                                                   Tuple[tf.Tensor, tf.Tensor]]:
    """
    Split the dataset for K-Fold validation.
    """

    kf = KFold(n_splits=k_folds, shuffle=True)

    train_k_folds_set = []
    test_k_folds_set = []
    for train_idx, test_idx in kf.split(data[0]):
        train_k_folds_set.append((tf.gather(data[0], train_idx), tf.gather(data[1], train_idx)))
        test_k_folds_set.append((tf.gather(data[0], test_idx), tf.gather(data[1], test_idx)))

    return train_k_folds_set, test_k_folds_set


def get_k_folds(data: Tuple[tf.Tensor, tf.Tensor], k_folds: int) -> [Tuple[tf.Tensor, tf.Tensor],
                                                                     Tuple[tf.Tensor, tf.Tensor]]:
    """
    Get k_folds data.
    """
    train_data, test_data = get_cross_validation_sets(data, k_folds=k_folds)

    return train_data, test_data


def wandb_log_prior_param(prior_params: dict, step: int):
    """
    Primarily to be used for logging in wandb the current prior params.
    """
    val_to_log = {}
    for k in prior_params.keys():
        val = prior_params[k][-1]
        val_to_log[f"learning-{k}"] = val
    wandb.log(val_to_log, step=step)
    logging.info(val_to_log)


def move_wandb_hydra_files(output_dir: str) -> str:
    """
    Move wandb and hydra files.
    """
    final_output_dir = f"outputs/{wandb.run.id}"
    # move hydra and wandb output files
    shutil.move(output_dir, final_output_dir)
    copy_tree(f"{os.path.sep}".join(wandb.run.dir.split(os.path.sep)[:-1]), os.path.join(final_output_dir))
    return final_output_dir


def bitmappify(ax, dpi=None):
    """
    Convert vector axes content to raster (bitmap) images
    """
    fig = ax.figure
    # safe plot without axes
    ax.set_axis_off()
    fig.savefig('temp.png', dpi=dpi, transparent=False)
    ax.set_axis_on()

    # remember geometry
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xb = ax.bbox._bbox.corners()[:,0]
    xb = (min(xb), max(xb))
    yb = ax.bbox._bbox.corners()[:,1]
    yb = (min(yb), max(yb))

    # compute coordinates to place bitmap image later
    xb = (- xb[0] / (xb[1] - xb[0]),
        (1 - xb[0]) / (xb[1] - xb[0]))
    xb = (xb[0] * (xl[1] - xl[0]) + xl[0],
        xb[1] * (xl[1] - xl[0]) + xl[0])
    yb = (- yb[0] / (yb[1] - yb[0]),
        (1 - yb[0]) / (yb[1] - yb[0]))
    yb = (yb[0] * (yl[1] - yl[0]) + yl[0],
        yb[1] * (yl[1] - yl[0]) + yl[0])

    # replace the dots by the bitmap
    del ax.collections[:]
    del ax.lines[:]
    ax.imshow(imread('temp.png'), origin='upper', aspect='auto', extent=(xb[0], xb[1], yb[0], yb[1]),
              label='_nolegend_')

    # reset view
    ax.set_xlim(xl)
    ax.set_ylim(yl)