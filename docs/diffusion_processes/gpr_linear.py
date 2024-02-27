import logging
import os

import gpflow.utilities.utilities
import numpy as np
import shutil

import tensorflow as tf
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call

from markovflow.models import GaussianProcessRegression
from markovflow.kernels import OrnsteinUhlenbeck
from docs.diffusion_processes.exp_dp_utils import get_hydra_output_dir, modify_time_grid, plot_posterior, calculate_nlpd, \
    setup_wandb, move_wandb_hydra_files, calculate_rmse


logger = logging.getLogger(__name__)


def get_model_nlpd(model, time_grid, test_obs):
    m, S = model.posterior.predict_y(time_grid)
    D = S.shape[-1]
    m = tf.reshape(m, (-1, D))
    S = tf.reshape(S, (-1, D, D))
    return calculate_nlpd(m, S, time_grid, test_obs)


def optimize_gpr(model: GaussianProcessRegression, lr: float, time_grid: tf.Tensor, test_obs: [tf.Tensor, tf.Tensor],
                 max_itr: int = 10000, optim_tol: float = 1e-2):
    opt = tf.optimizers.Adam(learning_rate=lr)

    @tf.function
    def opt_step():
        opt.minimize(model.loss, model.trainable_variables)

    loss_vals = [model.log_likelihood()]
    nlpd_vals = [get_model_nlpd(model, time_grid, test_obs)]
    if wandb.run:
        wandb.log({"log_likelihood": loss_vals[-1], "decay": model.kernel.decay.numpy(), "nlpd": nlpd_vals[-1]})

    for i in range(max_itr):
        opt_step()
        loss_vals.append(model.log_likelihood())
        nlpd_vals.append(get_model_nlpd(model, time_grid, test_obs))

        logger.info(f"Decay {i} iteration: {model.kernel.decay.numpy()}")

        if wandb.run:
            wandb.log({"log_likelihood": loss_vals[-1], "decay": model.kernel.decay.numpy(), "nlpd": nlpd_vals[-1]})

        if tf.math.abs(loss_vals[-2] - loss_vals[-1]) < optim_tol:
            logger.info("Model Optimized successfully!")
            break

    return loss_vals


@hydra.main(version_base="1.2", config_path="configs/", config_name="gpr_linear_process")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs = call(cfg.dataloader)

    if cfg.wandb.username is not None:
        if cfg.optimize:
            cfg.wandb.tags.append("Learning")
        else:
            cfg.wandb.tags.append("Inference")
        setup_wandb(cfg)

    logger.info(f"---------------------------------------------")
    logger.info(f"Starting with data path: {cfg.data_path}")
    logger.info(f"---------------------------------------------")
    logger.info(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    logger.info(f"---------------------------------------------")
    logger.info(f"Train data are {observations[0].shape[0]} and test data are {test_obs[0].shape[0]}")
    logger.info(f"---------------------------------------------")

    dt = time_grid[1] - time_grid[0]
    orig_time_grid = None
    if cfg.dt is not None and cfg.dt != dt:
        assert dt.numpy().item() > cfg.dt > 1e-6
        orig_time_grid = tf.identity(time_grid)
        time_grid = modify_time_grid(time_grid, cfg.dt)

    kernel = OrnsteinUhlenbeck(decay=cfg.decay, diffusion=Q)
    model = instantiate(cfg.model)(kernel=kernel, input_data=observations, chol_obs_covariance=noise_stddev)

    if cfg.optimize:
        gpflow.utilities.utilities.set_trainable(kernel.diffusion, False)
        optimize_gpr(model, lr=cfg.lr, optim_tol=cfg.optim_tol, max_itr=cfg.max_itr, time_grid=time_grid,
                     test_obs=test_obs)
        logger.info(f"Learnt kernel parameters :")
        logger.info(f"Decay : {kernel.decay.numpy()}")
        logger.info(f"Diffusion :{kernel.diffusion.numpy()}")

    logger.info(f"Log-likelihood : {model.log_likelihood()}")

    m, S = model.predict_f(time_grid)
    D = S.shape[-1]
    m = tf.reshape(m, (-1, D))
    S = tf.reshape(S, (-1, D, D))
    plot_posterior(m, S, observations[0], observations[1], time_grid, latent_process, orig_time_grid,
                   test_observations=test_obs, model_legend="GPR",
                   output_path=os.path.join(output_dir, "posterior.png"))

    # NLPD
    m_y, S_y = model.posterior.predict_y(time_grid)
    m_y = tf.reshape(m_y, (-1, D))
    S_y = tf.reshape(S_y, (-1, D, D))
    nlpd_val = calculate_nlpd(m_y, S_y, time_grid, test_obs)
    logger.info(f"Final NLPD on the test data: {nlpd_val}")

    # RMSE
    rmse_val = calculate_rmse(m_y, time_grid, test_obs)
    logger.info(f"Final RMSE on the test data: {rmse_val}")

    if wandb.run:
        if not cfg.optimize:
            wandb.log({"log_likelihood": model.log_likelihood(), "nlpd": nlpd_val, "rmse": rmse_val})
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    # Saving models and outputs
    np.savez(os.path.join(output_dir, "posteriors.npz"), gpr_m=m, gpr_S=S, time_grid=time_grid)
    np.savez(os.path.join(output_dir, "training_statistics.npz"), log_likelihood=model.log_likelihood(),
             nlpd=nlpd_val, rmse=rmse_val, decay=kernel.decay.numpy(), diffusion=kernel.diffusion.numpy())
    shutil.copy(cfg.data_path, os.path.join(output_dir, cfg.data_path.split("/")[-1]))


if __name__ == '__main__':
    run_experiment()
