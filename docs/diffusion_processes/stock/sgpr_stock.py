import logging
import os

import numpy as np
import shutil

from gpflow.models import SGPR
import gpflow
from gpflow.kernels import Constant, Linear, Matern12, Matern32
import tensorflow as tf
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call

from docs.diffusion_processes.exp_dp_utils import get_hydra_output_dir, modify_time_grid, plot_posterior, calculate_nlpd, \
    setup_wandb, move_wandb_hydra_files, calculate_rmse


logger = logging.getLogger(__name__)


def get_model_nlpd(model, time_grid, test_obs):
    m, S = model.predict_y(time_grid)
    D = m.shape[-1]
    m = tf.reshape(m, (-1, D))
    S = tf.reshape(S, (-1, D, D))
    return calculate_nlpd(m, S, time_grid, test_obs)


def optimize_gpr(model: SGPR, lr: float, time_grid: tf.Tensor, test_obs: [tf.Tensor, tf.Tensor],
                 max_itr: int = 10000, optim_tol: float = 1e-2):
    opt = tf.optimizers.Adam(learning_rate=lr)

    @tf.function
    def opt_step():
        opt.minimize(model.training_loss, model.trainable_variables)

    elbo_vals = [model.elbo()]
    nlpd_vals = [get_model_nlpd(model, time_grid, test_obs)]
    if wandb.run:
        wandb.log({"ELBO": elbo_vals[-1], "nlpd": nlpd_vals[-1]})

    for i in range(max_itr):
        opt_step()
        elbo_vals.append(model.elbo())
        nlpd_vals.append(get_model_nlpd(model, time_grid, test_obs))
        logger.info(f"ELBO: {elbo_vals[-1]}")
        logger.info(f"nlpd: {nlpd_vals[-1]}")

        if wandb.run:
            wandb.log({"ELBO": elbo_vals[-1], "nlpd": nlpd_vals[-1]})

        if tf.math.abs(elbo_vals[-2] - elbo_vals[-1]) < optim_tol:
            logger.info("Model Optimized successfully!")
            break

    return elbo_vals


@hydra.main(version_base="1.2", config_path="../configs/", config_name="sgpr_stock_process")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs = call(cfg.dataloader)
    time_grid = time_grid[..., None]
    observations = (observations[0][..., None], observations[1])
    test_obs = (test_obs[0][..., None], test_obs[1])

    if cfg.wandb.username is not None:
        if cfg.optimize:
            cfg.wandb.tags.append("Learning")
        else:
            cfg.wandb.tags.append("Inference")

        process_identifier = cfg.dataloader.data_path.split(os.path.sep)[-2]
        cfg.wandb.tags.append(process_identifier)
        setup_wandb(cfg)

    logger.info(f"---------------------------------------------")
    logger.info(f"Starting with data path: {cfg.data_path}")
    logger.info(f"---------------------------------------------")
    logger.info(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    logger.info(f"---------------------------------------------")
    logger.info(f"Train data are {observations[0].shape[0]} and test data are {test_obs[0].shape[0]}")
    logger.info(f"---------------------------------------------")

    kernel = Constant(variance=cfg.constant_kernel_variance) + Matern12(lengthscales=cfg.matern_lengthscale,
                                                                        variance=1.) + Linear(
        variance=cfg.linear_kernel_variance) + Matern32(lengthscales=cfg.matern_lengthscale, variance=1.)

    Z = tf.reshape(tf.linspace(time_grid[0], time_grid[-1], cfg.n_inducing_variables), (-1, 1))
    model = instantiate(cfg.model)(data=observations, kernel=kernel, inducing_variable=Z,
                                   noise_variance=noise_stddev**2)
    gpflow.utilities.set_trainable(model.inducing_variable, False)
    gpflow.utilities.set_trainable(model.likelihood.variance, False)
    gpflow.utilities.print_summary(model)

    if cfg.optimize:
        optimize_gpr(model, lr=cfg.lr, optim_tol=cfg.optim_tol, max_itr=cfg.max_itr, time_grid=time_grid,
                     test_obs=test_obs)

    logger.info(f"ELBO : {model.elbo()}")

    m, S = model.predict_f(time_grid)
    plot_posterior(m, S, observations[0], observations[1], time_grid, latent_process, time_grid,
                   test_observations=test_obs, model_legend="SGPR",
                   output_path=os.path.join(output_dir, "posterior.png"))

    m_y, S_y = model.predict_y(time_grid)
    nlpd_val = calculate_nlpd(m_y, S_y, time_grid, test_obs)
    logger.info(f"Final NLPD on the test data: {nlpd_val}")

    # RMSE
    rmse_val = calculate_rmse(m_y, time_grid, test_obs)
    logger.info(f"Final RMSE on the test data: {rmse_val}")

    if wandb.run:
        if not cfg.optimize:
            wandb.log({"ELBO": model.elbo(), "nlpd": nlpd_val, "rmse": rmse_val})
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    gpflow.utilities.print_summary(model)

    # Saving models and outputs
    np.savez(os.path.join(output_dir, "posteriors.npz"), gpr_m=m, gpr_S=S, time_grid=time_grid)
    np.savez(os.path.join(output_dir, "training_statistics.npz"), elbo=model.elbo(), nlpd=nlpd_val, rmse=rmse_val)
    tf.saved_model.save(model, os.path.join(output_dir, "sgpr_model"))

    shutil.copy(cfg.data_path, os.path.join(output_dir, cfg.data_path.split("/")[-1]))


if __name__ == '__main__':
    run_experiment()
