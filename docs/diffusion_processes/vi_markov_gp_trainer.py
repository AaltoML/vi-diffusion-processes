"""
Trainer for a VI Gaussian Approximation, Archambeau et al.
"""
import logging

import wandb
import tensorflow as tf
from tensorflow_probability import distributions

from markovflow.sde import OrnsteinUhlenbeckSDE, MLPDrift
from markovflow.models.vi_sde import VariationalMarkovGP
from docs.diffusion_processes.exp_dp_utils import calculate_nlpd, wandb_log_prior_param, calculate_rmse

logger = logging.getLogger(__name__)


class VIMarkovGPTrainer:
    """
    Class that trains the VariationalMarkovGP model and encapsulates all the other details.
    """

    def __init__(self, model: VariationalMarkovGP, test_data: [tf.Tensor, tf.Tensor],
                 q_lr: float = 0.1, x0_lr: float = 0.1, max_itr: int = 100, lr_tol: float = 1e-2,
                 optim_tol: float = 1e-2, warmup_x0_itr: int = 10, learn_prior_sde: bool = False,
                 prior_sde_lr: float = 1e-2, learning_max_itr: int = 100, learning_tol: float = 1e-2,
                 optimize_prior_initial_state: bool = False, prior_initial_state_lr: float = 0.1
                 ):
        self.model = model
        self.test_data = test_data
        self.max_itr = max_itr
        self.q_lr = q_lr
        self.x0_lr = x0_lr
        self.lr_tol = lr_tol
        self.optim_tol = optim_tol
        self.warmup_x0_itr = warmup_x0_itr
        self.learn_prior_sde = learn_prior_sde
        self.prior_sde_lr = prior_sde_lr
        self.learning_max_itr = learning_max_itr
        self.learning_tol = learning_tol
        self.optimize_prior_initial_state = optimize_prior_initial_state
        self.prior_initial_state_lr = prior_initial_state_lr
        self.prior_params = {}
        self.n_prior_step = 0
        self.n_optimize_step = 0

        if self.learn_prior_sde:
            self.prior_sde_optim = tf.optimizers.Adam(learning_rate=self.prior_sde_lr)
            self.store_prior_param_vals()

    def perform_inference(self):
        """
        Perform inference till convergence
        """
        elbo_vals_inf = [self.model.elbo()]   # Added for comparison, while returning removed the first one to avoid double counting
        nlpd_vals_inf = []
        rmse_vals_inf = []

        # Done this way because we need to reset them for next learning iteration
        q_lr = self.q_lr
        x0_lr = self.x0_lr

        for i in range(self.max_itr):
            m, S = self.model.forward_pass
            self.model.update_lagrange(m, S)
            self.model.update_param(m, S, lr=q_lr)

            if i > self.warmup_x0_itr:
                self.model.update_initial_statistics(lr=x0_lr)

            m, S = self.model.forward_pass
            elbo_vals_inf.append(self.model.elbo(m, S))
            nlpd_vals_inf.append(self.model_nlpd(m, S))
            rmse_vals_inf.append(self.model_rmse(m, S))

            self.n_optimize_step = self.n_optimize_step + 1
            if wandb.run:
                wandb.log({"elbo": elbo_vals_inf[-1], "nlpd": nlpd_vals_inf[-1], "rmse": rmse_vals_inf[-1]},
                          step=self.n_optimize_step)

            if (elbo_vals_inf[-2] > elbo_vals_inf[-1]) or (
                    tf.math.abs(elbo_vals_inf[-2] - elbo_vals_inf[-1]) < self.lr_tol):
                logger.info("Decaying lr!!!")
                q_lr = q_lr / 10
                x0_lr = x0_lr / 10

            if tf.math.abs(elbo_vals_inf[-2] - elbo_vals_inf[-1]) < self.optim_tol:
                logger.info("Model successfully optimized!!!")
                break

            logger.info(f"ELBO {i + 1}: {elbo_vals_inf[-1]}")

        return elbo_vals_inf[1:], nlpd_vals_inf, rmse_vals_inf

    def optimize(self):
        """
        Optimize the VI Markov model.
        """

        # Warmup iterations
        for _ in range(20):
            m, S = self.model.forward_pass
            self.model.update_lagrange(m, S)
            self.model.update_param(m, S, lr=1e-6)
            print(f"Warmup : {self.model.elbo()}")

        m, S = self.model.forward_pass
        elbo_vals = [self.model.elbo(m, S)]
        nlpd_vals = [self.model_nlpd(m, S)]
        rmse_vals = [self.model_rmse(m, S)]

        if wandb.run:
            wandb.log({"elbo": elbo_vals[-1], "nlpd": nlpd_vals[-1], "rmse": rmse_vals[-1]}, step=self.n_optimize_step)
            if self.learn_prior_sde:
                wandb_log_prior_param(self.prior_params, step=self.n_prior_step)
        # loop for learning
        for _ in range(self.max_itr):
            elbo_vals_inf, nlpd_vals_inf, rmse_vals_inf = self.perform_inference()
            elbo_vals = elbo_vals + elbo_vals_inf
            nlpd_vals = nlpd_vals + nlpd_vals_inf
            rmse_vals = rmse_vals + rmse_vals_inf

            if self.learn_prior_sde:
                prior_learning_elbo, prior_learning_nlpd, prior_learning_rmse = self.optimize_prior_sde()

                elbo_vals = elbo_vals + prior_learning_elbo
                nlpd_vals = nlpd_vals + prior_learning_nlpd
                rmse_vals = rmse_vals + prior_learning_rmse
            else:
                break  # no learning and inference has converged

            if (len(prior_learning_elbo) > 2 and (tf.math.abs(prior_learning_elbo[-2] - prior_learning_elbo[-1]))) < self.optim_tol:
                logger.info("Model successfully optimized!!!")
                break

        return elbo_vals, nlpd_vals, rmse_vals, self.prior_params

    def model_nlpd(self, m, S):
        """
        Get model NLPD
        """
        m, S = self.model.likelihood.predict_mean_and_var(m, S)
        # S = S[:, :, 0]
        return calculate_nlpd(m, S, self.model.grid, self.test_data)

    def model_rmse(self, m, S):
        """
        Get model NLPD
        """
        m, S = self.model.likelihood.predict_mean_and_var(m, S)
        return calculate_rmse(m, self.model.grid, self.test_data)

    def store_prior_param_vals(self):
        """Update the list storing the prior sde parameter values"""
        if isinstance(self.model.prior_sde, MLPDrift):
            return
        for i, param in enumerate(self.model.prior_sde.trainable_variables):
            i = str(i)  # Store key as str. used later as string to save.
            if i in self.prior_params.keys():
                self.prior_params[i].append(param.numpy().item())
            else:
                self.prior_params[i] = [param.numpy().item()]

    def optimize_prior_sde(self):
        """
        Optimize the prior sde.
        """
        elbo_vals = [self.model.elbo()]
        nlpd_vals = []
        rmse_vals = []

        logger.info("Learning prior parameters...")
        for i in range(self.learning_max_itr):
            grads = self.model.grad_prior_sde_params()
            self.prior_sde_optim.apply_gradients(
                [(g, v) for g, v in zip(grads, self.model.prior_sde.trainable_variables)])

            if self.optimize_prior_initial_state:
                self.optimize_prior_x0()

            m, S = self.model.forward_pass
            elbo_vals.append(self.model.elbo(m, S))
            nlpd_vals.append(self.model_nlpd(m, S))
            rmse_vals.append(self.model_rmse(m, S))

            self.store_prior_param_vals()
            self.n_prior_step = self.n_prior_step + 1
            if wandb.run:
                wandb_log_prior_param(self.prior_params, step=self.n_prior_step)

            if wandb.run:
                wandb.log({"elbo": elbo_vals[-1], "nlpd": nlpd_vals[-1], "rmse": rmse_vals[-1]},
                          step=self.n_optimize_step)
                self.n_optimize_step = self.n_optimize_step + 1

            logger.info(f"ELBO value at {i + 1}: {elbo_vals[-1]}")

            if tf.math.abs(elbo_vals[-2] - elbo_vals[-1]) < self.learning_tol:
                logger.info("Prior parameter optimized successfully!!!")
                break

        return elbo_vals[1:], nlpd_vals, rmse_vals

    def optimize_prior_x0(self):
        """
        Optimize the initial state of the prior
        """

        if isinstance(self.model.prior_sde, OrnsteinUhlenbeckSDE):
            p_initial_mean = self.model.prior_initial_state.loc
            p_initial_chol_cov = self.model.prior_sde.q/(2*self.model.prior_sde.decay) + 0 * self.model.prior_initial_state.scale.tril

        else:
            grads = self.model.grad_initial_state()
            p_initial_mean = self.model.prior_initial_state.loc - self.prior_initial_state_lr * grads[0]
            p_initial_chol_cov = self.model.prior_initial_state.scale.tril - self.prior_initial_state_lr * grads[1]

        self.model.prior_initial_state = distributions.MultivariateNormalFullCovariance(loc=p_initial_mean,
                                                                                        covariance_matrix=p_initial_chol_cov @ p_initial_chol_cov)

