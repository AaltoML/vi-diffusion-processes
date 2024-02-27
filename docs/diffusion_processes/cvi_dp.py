import logging
from datetime import datetime
import os

import numpy as np
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from gpflow.probability_distributions import Gaussian
from markovflow.sde import OrnsteinUhlenbeckSDE, DoubleWellSDE, MLPDrift
from markovflow.sde.sde_utils import euler_maruyama
from markovflow.likelihoods import MultivariateGaussian

from docs.diffusion_processes.exp_dp_utils import get_hydra_output_dir, modify_time_grid, plot_posterior, \
    plot_line, setup_wandb, move_wandb_hydra_files

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs/", config_name="cvi_linear_process")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs = call(cfg.dataloader)
    state_dim = observations[1].shape[-1]
    learn_prior = cfg.trainer.learn_prior_sde

    if cfg.wandb.username is not None:
        if learn_prior:
            cfg.wandb.tags.append("Learning")
        else:
            cfg.wandb.tags.append("Inference")
        cfg.wandb.tags.append(cfg.prior_sde._target_.split(".")[-1])
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
        dt = time_grid[1] - time_grid[0]

    prior_sde = instantiate(cfg.prior_sde)(q=Q * tf.eye(state_dim, dtype=observations[1].dtype))
    if isinstance(prior_sde, OrnsteinUhlenbeckSDE):
        prior_sde.q = Q * tf.eye(state_dim, dtype=observations[1].dtype)
        steady_cov = Q / (2 * prior_sde.decay)
        initial_state = Gaussian(mu=tf.zeros((state_dim,), dtype=time_grid.dtype),
                                 cov=steady_cov * tf.eye(state_dim, dtype=time_grid.dtype))

    else:
        prior_sde.q = tf.ones((state_dim, state_dim), dtype=observations[1].dtype) * Q
        initial_state = Gaussian(mu=cfg.prior_x0_mu * tf.ones((state_dim,), dtype=time_grid.dtype),
                                 cov=cfg.prior_x0_cov * tf.eye(state_dim, dtype=time_grid.dtype))

    model = instantiate(cfg.model)(prior_sde, time_grid, observations, prior_initial_state=initial_state,
                                   stabilize_ssm=cfg.stabilize_ssm, clip_state_transitions=cfg.clip_state_transitions)

    # Likelihood variance
    model.likelihood = MultivariateGaussian(tf.eye(state_dim, dtype=observations[1].dtype) * noise_stddev.item())

    start_time = datetime.now()
    trainer = instantiate(cfg.trainer)(model=model, prior_sde=prior_sde, test_data=test_obs)
    elbo_vals, nlpd_vals, rmse_vals, prior_params = trainer.optimize()
    end_time = datetime.now()

    logger.info(f"ELBO value after optimization : {elbo_vals[-1]}")
    logger.info(f"Time taken : {end_time - start_time}")

    m, S = model.dist_q.marginals
    plot_posterior(m, S, observations[0], observations[1], time_grid, latent_process,
                   latent_process_grid=orig_time_grid, test_observations=test_obs,
                   model_legend=f"Proposed Model (dt={dt})",
                   output_path=os.path.join(output_dir, "posterior.png"))
    plot_line(elbo_vals, output_path=os.path.join(output_dir, "elbo.png"), title="ELBO")
    plot_line(nlpd_vals, output_path=os.path.join(output_dir, "nlpd.png"), title="NLPD")
    plot_line(rmse_vals, output_path=os.path.join(output_dir, "rmse.png"), title="RMSE")

    logger.info(f"Final NLPD on the test data: {nlpd_vals[-1]}")
    logger.info(f"Final RMSE on the test data: {rmse_vals[-1]}")

    if learn_prior:
        if len(prior_params.keys()) == 1:
            plot_line(trainer.prior_params["0"], output_path=os.path.join(output_dir, "learnt_prior_params.png"),
                      title="Learnt prior param")
        else:
            if isinstance(model.prior_sde, DoubleWellSDE):
                x = np.linspace(-1.5, 1.5, 100).reshape((-1, 1))
            elif "GPS" in cfg.wandb.tags:
                x = np.linspace(-15, 15, 100).reshape((-1, 1))
            elif "Apple Stock" in cfg.wandb.tags:
                x = np.linspace(-6, 6, 100).reshape((-1, 1))
            else:
                x = np.linspace(-2, 2, 100).reshape((-1, 1))

            drift_val = model.prior_sde.drift(x, t=None)
            plt.clf()
            plt.plot(x, drift_val)
            plt.xlabel("Iterations")
            plt.title("Learnt Drift")

            if wandb.run is not None:
                wandb.log({"Learnt Drift": wandb.Image(plt)})

            plt.show()

        # sample from a learnt prior_sde
        sample = euler_maruyama(model.prior_sde, x0 * tf.ones((1, 1), dtype=time_grid.dtype), time_grid)[0]
        plt.subplots(1, 1, figsize=(15, 5))
        plt.plot(time_grid, sample)
        plt.title("Learnt Prior sample")
        if wandb.run is not None:
            wandb.log({"Learnt prior sample": wandb.Image(plt)})
        plt.show()

        logger.info(f"Learnt prior parameters are:")
        for k in trainer.prior_params.keys():
            logger.info(f"{k}: {trainer.prior_params[k][-1]}")

    if wandb.run:
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    # Saving models and outputs
    np.savez(os.path.join(output_dir, "posteriors.npz"), cvi_m=m, cvi_S=S, time_grid=time_grid)
    np.savez(os.path.join(output_dir, "training_statistics.npz"), elbo=elbo_vals, nlpd=nlpd_vals, rmse=rmse_vals)
    shutil.copy(cfg.data_path, os.path.join(output_dir, cfg.data_path.split("/")[-1]))

    np.savez(os.path.join(output_dir, "cvi_model.npz"), data_sites_nat1=model.data_sites.nat1.numpy(),
             data_sites_nat2=model.data_sites.nat2.numpy(),
             girsanov_sites_nat1=model.girsanov_sites.nat1.numpy(),
             girsanov_sites_nat2_diag=model.girsanov_sites.nat2.block_diagonal.numpy(),
             girsanov_sites_nat2_subdiag=model.girsanov_sites.nat2.block_sub_diagonal.numpy()
             )

    if learn_prior:
        if isinstance(model.prior_sde, MLPDrift):
            model.prior_sde.MLP.save_weights(os.path.join(output_dir, "learnt_prior_mlp"))
        else:
            np.savez(os.path.join(output_dir, "learnt_prior_params.npz"), **prior_params)


if __name__ == '__main__':
    run_experiment()
