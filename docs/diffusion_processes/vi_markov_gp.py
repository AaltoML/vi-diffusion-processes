import logging
from datetime import datetime
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from markovflow.sde.drift import LinearDrift
import matplotlib.pyplot as plt

from markovflow.likelihoods import MultivariateGaussian
from markovflow.sde import OrnsteinUhlenbeckSDE, DoubleWellSDE, MLPDrift
from docs.diffusion_processes.exp_dp_utils import get_hydra_output_dir, modify_time_grid, plot_posterior, plot_line, \
    setup_wandb, move_wandb_hydra_files

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs/", config_name="vi_linear_process")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    Q, x0, noise_stddev, latent_process, observations, time_grid, test_obs = call(cfg.dataloader)

    state_dim = observations[1].shape[-1]
    learn_prior = cfg.trainer.learn_prior_sde

    if cfg.wandb.username:
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

    if cfg.prior_x0_mean is not None and cfg.prior_x0_cov is not None:
        loc = cfg.prior_x0_mean * tf.ones((state_dim,), dtype=time_grid.dtype)
        cov = cfg.prior_x0_cov * tf.eye(state_dim, dtype=time_grid.dtype)

        prior_x0 = distributions.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
    elif cfg.prior_x0_mean is not None and cfg.prior_x0_cov is None:
        loc = cfg.prior_x0_mean * tf.ones((state_dim,), dtype=time_grid.dtype)
        cov = Q * tf.eye(state_dim, dtype=time_grid.dtype)
        prior_x0 = distributions.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
    elif cfg.prior_x0_mean is None and cfg.prior_x0_cov is not None:
        raise Exception("When providing prior on initial state both mean and covariance should be provided!!!")
    else:
        prior_x0 = None

    model = instantiate(cfg.model)(prior_sde=prior_sde, grid=time_grid, input_data=observations,
                                   prior_initial_state=prior_x0, stabilize_system=cfg.stabilize_system)

    model.likelihood = MultivariateGaussian(tf.eye(state_dim, dtype=observations[1].dtype) * noise_stddev.item())

    # Helping model a bit in linear case
    if isinstance(prior_sde, OrnsteinUhlenbeckSDE):
        model.A = prior_sde.decay + 0. * model.A

        p_initial_cov = (Q/(2*prior_sde.decay)) * tf.eye(state_dim, dtype=time_grid.dtype)
        model.prior_initial_state = distributions.MultivariateNormalFullCovariance(loc=model.prior_initial_state.loc,
                                                                                   covariance_matrix=p_initial_cov)

    if cfg.cvi_posterior_path is not None:
        cvi_model_data = np.load(os.path.join(cfg.cvi_posterior_path, "cvi_model.npz"))
        cvi_model_posterior = np.load(os.path.join(cfg.cvi_posterior_path, "posteriors.npz"))

        from markovflow.models import CVISitesSDE
        from gpflow.probability_distributions import Gaussian as G_pd
        from gpflow.likelihoods import Gaussian

        initial_posterior_path = G_pd(mu=tf.cast(cvi_model_posterior["cvi_m"], dtype=time_grid.dtype),
                                      cov=tf.cast(cvi_model_posterior["cvi_S"], dtype=time_grid.dtype))
        prior_initial_state = G_pd(mu=prior_x0.loc, cov=prior_x0.scale**2)

        cvi_model = CVISitesSDE(prior_sde, time_grid, observations, Gaussian(noise_stddev.item() ** 2),
                                prior_initial_state=prior_initial_state,
                                initial_posterior_path=initial_posterior_path, stabilize_ssm=False)
        cvi_model.data_sites.nat1.assign(cvi_model_data["data_sites_nat1"])
        cvi_model.data_sites.nat2.assign(cvi_model_data["data_sites_nat2"])
        cvi_model.girsanov_sites.nat1.assign(cvi_model_data["girsanov_sites_nat1"])
        cvi_model.girsanov_sites.nat2.block_diagonal.assign(cvi_model_data["girsanov_sites_nat2_diag"])
        cvi_model.girsanov_sites.nat2.block_sub_diagonal.assign(cvi_model_data["girsanov_sites_nat2_subdiag"])

        dist_q_ssm = cvi_model.dist_q
        drift = LinearDrift()
        drift.set_from_ssm(dist_q_ssm, time_grid[1] - time_grid[0])

        model.A = model.A * 0 - drift.A
        model.b = model.b * 0 + drift.b
    elif cfg.vi_model_path is not None:
        vi_model_data = np.load(os.path.join(cfg.vi_model_path, "vi_gp_model.npz"))
        model.A = model.A * 0 + vi_model_data["A"]
        model.b = model.b * 0 + vi_model_data["b"]
        model.lambda_lagrange = model.lambda_lagrange * 0 + vi_model_data["lambda_lagrange"]
        model.psi_lagrange = model.psi_lagrange * 0 + vi_model_data["psi_lagrange"]

    start_time = datetime.now()
    trainer = instantiate(cfg.trainer)(model=model, test_data=test_obs)
    elbo_vals, nlpd_vals, rmse_vals, prior_params = trainer.optimize()
    end_time = datetime.now()

    logger.info(f"ELBO value after optimization : {elbo_vals[-1]}")
    logger.info(f"Time taken : {end_time - start_time}")

    m, S = model.forward_pass
    plot_posterior(m, S, observations[0], observations[1], time_grid, latent_process, orig_time_grid,
                   test_observations=test_obs, model_legend=f"Archambeau et al. (dt={dt}) ",
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

    if wandb.run:
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    # Saving models and outputs
    np.savez(os.path.join(output_dir, "posteriors.npz"), vi_m=m, vi_S=S, time_grid=time_grid)
    np.savez(os.path.join(output_dir, "training_statistics.npz"), elbo=elbo_vals, nlpd=nlpd_vals, rmse=rmse_vals)
    shutil.copy(cfg.data_path, os.path.join(output_dir, cfg.data_path.split("/")[-1]))

    np.savez(os.path.join(output_dir, "vi_gp_model.npz"), A=model.A.numpy(),
             b=model.b.numpy(), lambda_lagrange=model.lambda_lagrange.numpy(),
             psi_lagrange=model.psi_lagrange.numpy(), x0_m=model.q_initial_state.loc,
             x0_S=model.q_initial_state.scale_tril @ tf.transpose(model.q_initial_state.scale_tril))

    if isinstance(model.prior_sde, MLPDrift):
        model.prior_sde.MLP.save_weights(os.path.join(output_dir, "learnt_prior_mlp"))
    else:
        np.savez(os.path.join(output_dir, "learnt_prior_params.npz"), **prior_params)


if __name__ == '__main__':
    run_experiment()
