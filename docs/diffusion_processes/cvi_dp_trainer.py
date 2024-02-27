"""
Trainer for a CVISitesSSM model.
"""
from typing import Union
import logging
import wandb

import tensorflow as tf

from markovflow.models import CVISitesSSM, CVISitesSDE
from markovflow.sde import SDE, MLPDrift, OrnsteinUhlenbeckSDE
from markovflow.sde.sde_utils import tranform_girsanov_sites
from docs.diffusion_processes.exp_dp_utils import calculate_nlpd, wandb_log_prior_param, calculate_rmse
from gpflow.probability_distributions import Gaussian

logger = logging.getLogger(__name__)


class CVISitesTrainer:
    """
       Class that trains the VariationalMarkovGP model and encapsulates all the other details.
    """

    def __init__(self, model: Union[CVISitesSSM, CVISitesSDE], test_data: [tf.Tensor, tf.Tensor], prior_sde: SDE,
                 max_itr: int = 100, optim_tol: float = 1e-2, max_itr_sites_optim: int = 20,
                 girsanov_sites_lr: float = 0.1, data_sites_lr: float = 0.1, learn_prior_sde: bool = False,
                 prior_sde_lr: float = 1e-2, learning_max_itr: int = 100, learning_tol: float = 1e-2):
        """
        If no linearization is required and model should use the same prior ssm then pass prior sde as None.
        """
        self.model = model
        self.prior_sde = prior_sde
        self.test_data = test_data
        self.max_itr = max_itr
        self.optim_tol = optim_tol
        self.max_itr_sites_optim = max_itr_sites_optim
        self.girsanov_sites_lr = girsanov_sites_lr
        self.data_sites_lr = data_sites_lr
        self.learn_prior_sde = learn_prior_sde
        self.prior_sde_lr = prior_sde_lr
        self.prior_params = {}
        self.learning_max_itr = learning_max_itr
        self.learning_tol = learning_tol
        self.n_prior_step = 0
        self.n_optimize_step = 0
        if self.learn_prior_sde:
            self.prior_sde_optim = tf.optimizers.Adam(learning_rate=self.prior_sde_lr)
            self.store_prior_param_vals()
            if wandb.run:
                wandb_log_prior_param(self.prior_params, self.n_prior_step)

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

    def _optimize_sites_under_stable_prior(self):
        """
        Optimize the sities till convergence under the current stable prior.
        """
        tmp_elbo_vals = [self.model.classic_elbo()]
        tmp_nlpd_vals = []
        tmp_rmse_vals = []

        while (len(tmp_elbo_vals) - 1) < self.max_itr_sites_optim:  # -1 as tmp_elbo already has one value
            self.model.update_data_sites(self.data_sites_lr)
            self.model.update_girsanov_sites(self.girsanov_sites_lr)

            tmp_elbo_vals.append(self.model.classic_elbo())
            tmp_nlpd_vals.append(self.model_nlpd())
            tmp_rmse_vals.append(self.model_rmse())

            if wandb.run:
                wandb.log({"elbo": tmp_elbo_vals[-1], "nlpd": tmp_nlpd_vals[-1], "rmse": tmp_rmse_vals[-1]},
                          step=self.n_optimize_step)
                self.n_optimize_step = self.n_optimize_step + 1

            if len(tmp_elbo_vals) > 2 and tmp_elbo_vals[-2] > tmp_elbo_vals[-1]:
                logger.info("Decaying LR! ELBO decreasing!!!")
                self.girsanov_sites_lr = self.girsanov_sites_lr / 10
                self.data_sites_lr = self.data_sites_lr / 10

            if len(tmp_elbo_vals) > 2 and tf.abs(tmp_elbo_vals[-2] - tmp_elbo_vals[-1]) < self.optim_tol:
                logger.info("Breaking the site updates loop. ELBO converged!")
                break

            logger.info(f"ELBO -{len(tmp_elbo_vals) - 1} : {tmp_elbo_vals[-1]}")

        return tmp_elbo_vals[1:], tmp_nlpd_vals, tmp_rmse_vals

    def perform_inference(self):
        """
        Perform inference till convergence
        """
        elbo_vals = [self.model.classic_elbo()]  # Added for comparison, while returning removed the first one to avoid double counting
        nlpd_vals = []
        rmse_vals = []

        for i_inference in range(self.max_itr):
            elbo_before_optim = self.model.classic_elbo()
            tmp_elbo_vals, tmp_nlpd_vals, tmp_rmse_vals = self._optimize_sites_under_stable_prior()

            # Transform sites under p_{lin}
            self.model.girsanov_sites = tranform_girsanov_sites(self.model.girsanov_sites, self.model.dist_p,
                                                                self.model.dist_p_linearized)
            self.model.dist_p = self.model.dist_p_linearized
            elbo_after_optim = self.model.classic_elbo()

            if tf.abs(elbo_before_optim - elbo_after_optim) < self.optim_tol:
                logger.info("ELBO converged! Optimization successully completed!")
                elbo_vals = elbo_vals + tmp_elbo_vals
                nlpd_vals = nlpd_vals + tmp_nlpd_vals
                rmse_vals = rmse_vals + tmp_rmse_vals
                return elbo_vals[1:], nlpd_vals, rmse_vals

            elbo_vals = elbo_vals + tmp_elbo_vals
            nlpd_vals = nlpd_vals + tmp_nlpd_vals
            rmse_vals = rmse_vals + tmp_rmse_vals

            # Don't perform linearization for last loop as then you won't update sites using this linearized prior
            if i_inference != (self.max_itr - 1):
                logger.info(f"Re-linearizing...")
                dist_p_last = self.model.dist_p  # sites are already transformed to the original p_L
                self.model.set_linearized_prior()
                logger.info(f"Linearization done..")
                # Transform sites to new linearized prior
                self.model.girsanov_sites = tranform_girsanov_sites(self.model.girsanov_sites, dist_p_last,
                                                                    self.model.dist_p)

        return elbo_vals[1:], nlpd_vals, rmse_vals

    def optimize(self) -> [list, list]:
        """
        Optimize sites of the model. As this is for linear case we can just do one single step update.

        Returns a list of elbo vals
        """
        elbo_vals = [self.model.classic_elbo()]
        nlpd_vals = [self.model_nlpd()]
        rmse_vals = [self.model_rmse()]

        set_of_previous_inf_vals = []  # while learning the two objective keeps on jumping between two vals to compare with a set of previous vals

        if wandb.run:
            wandb.log({"elbo": elbo_vals[0], "nlpd": nlpd_vals[0], "rmse": rmse_vals[0]}, step=self.n_optimize_step)
            self.n_optimize_step = self.n_optimize_step + 1

        for i_learning in range(self.max_itr):  # Loop for learning

            inf_elbo_vals, inf_nlpd_vals, inf_rmse_vals = self.perform_inference()
            elbo_vals = elbo_vals + inf_elbo_vals
            nlpd_vals = nlpd_vals + inf_nlpd_vals
            rmse_vals = rmse_vals + inf_rmse_vals

            if self.learn_prior_sde:
                prior_learning_elbo_vals, prior_learning_nlpd_vals, prior_learning_rmse_vals = self.optimize_prior_sde()
            else:
                break  # No learning and inference has converged

            if tf.math.abs(elbo_vals[-1] - prior_learning_elbo_vals[-1]) < self.optim_tol:
                elbo_vals = elbo_vals + prior_learning_elbo_vals
                nlpd_vals = nlpd_vals + prior_learning_nlpd_vals
                rmse_vals = rmse_vals + prior_learning_rmse_vals
                logger.info("Model successfully optimized!!!")
                break

            elbo_vals = elbo_vals + prior_learning_elbo_vals
            nlpd_vals = nlpd_vals + prior_learning_nlpd_vals
            rmse_vals = rmse_vals + prior_learning_rmse_vals

            # To hold the zigzag pattern and break the loop
            if len(set_of_previous_inf_vals) > 4:
                if (tf.math.abs(set_of_previous_inf_vals[-1] - set_of_previous_inf_vals[-3]) < 1e-4) or \
                        (tf.math.abs(set_of_previous_inf_vals[-2] - set_of_previous_inf_vals[-4]) < 1e-4):
                    logger.info("The objective is most probably jumping between two values!!!")
                    logger.info("Model successfully optimized!!!")
                    break

            set_of_previous_inf_vals.append(elbo_vals[-1])

        return elbo_vals, nlpd_vals, rmse_vals, self.prior_params

    def model_nlpd(self):
        """
        Get model NLPD
        """
        m, S = self.model.dist_q.marginals
        m, S = self.model.likelihood.predict_mean_and_var(m, S)
        # S = S[:, :, 0]
        return calculate_nlpd(m, S, self.model.time_grid, self.test_data)

    def model_rmse(self):
        """
        Get model RMSE
        """
        m, S = self.model.dist_q.marginals
        m, S = self.model.likelihood.predict_mean_and_var(m, S)

        return calculate_rmse(m, self.model.time_grid, self.test_data)

    def optimize_prior_sde(self):
        """
        Optimize the prior sde.
        """
        elbo_vals = [self.model.classic_elbo()]
        nlpd_vals = []
        rmse_vals = []
        logger.info("Learning prior parameters...")
        for i in range(self.learning_max_itr):
            grads_kl = self.model.grad_KL_wrt_prior_params()
            grads_ve = self.model.grad_VE_wrt_prior_params()
            grads = [sum(x) for x in zip(grads_kl, grads_ve)]
            self.prior_sde_optim.apply_gradients(
                [(g, v) for g, v in zip(grads, self.model.prior_sde.trainable_variables)])
            elbo_vals.append(self.model.classic_elbo())
            nlpd_vals.append(self.model_nlpd())
            rmse_vals.append(self.model_rmse())

            self.n_prior_step += 1
            self.store_prior_param_vals()
            if wandb.run:
                wandb_log_prior_param(self.prior_params, self.n_prior_step)

            if isinstance(self.model.prior_sde, OrnsteinUhlenbeckSDE):
                steady_cov = self.model.prior_sde.q / (2 * self.model.prior_sde.decay)
                self.model.prior_initial_state = Gaussian(mu=tf.zeros((1,), dtype=self.model.time_grid.dtype),
                                                          cov=steady_cov * tf.ones((1, 1),
                                                                                   dtype=self.model.time_grid.dtype))

            logger.info(f"ELBO value at {i + 1}: {elbo_vals[-1]}")
            if wandb.run:
                wandb.log({"elbo": elbo_vals[-1], "nlpd": nlpd_vals[-1], "rmse": rmse_vals[-1]},
                          step=self.n_optimize_step)
                self.n_optimize_step = self.n_optimize_step + 1

            if elbo_vals[-1] < elbo_vals[-2]:
                logger.info("Decaying the LR!!!")
                self.prior_sde_optim.lr = self.prior_sde_optim.lr / 2

            if tf.math.abs(elbo_vals[-2] - elbo_vals[-1]) < self.learning_tol:
                logger.info("Prior parameter optimized successfully!!!")
                break

        return elbo_vals[1:], nlpd_vals, rmse_vals
