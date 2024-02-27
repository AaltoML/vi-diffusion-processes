#
# Copyright (c) 2021 The Markovflow Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
    Module containing a model for Variational-CVI in diffusion processes.

    @inproceedings{verma2024cvidp,
        title={Variational Inference for Diffusion Processes},
        author={Verma, Prakhar and Adam, Vincent and Solin, Arno},
        booktitle={International Conference on Artificial Intelligence and Statistics},
        year={2024},
        organization={PMLR}
}

"""
from abc import ABC
from typing import Tuple

import tensorflow as tf
from gpflow.likelihoods import Likelihood
from gpflow.base import Parameter, TensorType
from gpflow.probability_distributions import Gaussian
import tensorflow_probability as tfp

from markovflow.sde import SDE
from markovflow.models import MarkovFlowModel
from markovflow.posterior import PosteriorProcess
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import ssm_to_btd_nat, SSM_KL_along_Gaussian_path, SSM_KL_with_grads_wrt_exp_params, \
    linearize_sde, remove_batch_from_ssm, SDE_SSM_KL_with_grads_wrt_exp_params
from markovflow.models.variational_cvi import gradient_transformation_mean_var_to_expectation
from markovflow.ssm_gaussian_transformations import naturals_to_ssm_params
from markovflow.kalman_filter import GaussianSitesNat
from markovflow.gauss_markov import BTDGaussian


class CVISitesSSM(MarkovFlowModel):
    """
    Provides a site-based parameterization to the variational posterior over the state trajectory of a SSM.
    ELBO of the model is defined as:
                    E_{q}[log p(Y|X)] - KL[q||p]
    """

    def __init__(
            self,
            prior_ssm: StateSpaceModel,
            time_grid: TensorType,
            input_data: Tuple[TensorType, TensorType],
            likelihood: Likelihood,
            prior_initial_state: Gaussian = None,
            initial_posterior_path: Gaussian = None,
    ) -> None:
        """
        :param prior_ssm: Prior SSM over the latent states, x.
        :param time_grid: Grid over time with shape ``batch_shape + [grid_size]``
        :param input_data: A tuple containing the observed data:
            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``
        :param likelihood: A likelihood for the observations of the model.
        :param prior_initial_state: A Gaussian prior on the initial state.
        :param initial_posterior_path: A Gaussian prior on the initial posterior path.

        Note: Currently, batching isn't supported!
        """
        super().__init__()

        if prior_ssm is not None:
            assert prior_ssm.batch_shape.is_compatible_with(
                tf.TensorShape([])), "CVISitesSSM model currently does not support batch!"

        self.likelihood = likelihood
        self.time_grid = time_grid
        self.dt = float(self.time_grid[1] - self.time_grid[0])
        self.dist_p = prior_ssm

        self._observations_time_points = input_data[0]
        self._observations = input_data[1]

        # We keep the both state and observations state same
        self.output_dim = self._observations.shape[-1]
        self.state_dim = self._observations.shape[-1]

        self.init_girsanov_sites()

        self.data_sites = GaussianSitesNat(
            nat1=Parameter(tf.zeros((self._observations_time_points.shape[0], self.state_dim),
                                    dtype=self._observations.dtype)),
            nat2=Parameter(tf.eye(self.state_dim, batch_shape=(self._observations_time_points.shape[0], ),
                                  dtype=self._observations.dtype) * 1e-10),
            log_norm=Parameter(tf.zeros((self._observations_time_points.shape[0], self.state_dim),
                                        dtype=self._observations.dtype))
        )

        # Initialize the prior on the initial state
        self.prior_initial_state = self._initialize_initial_state_prior(prior_initial_state)

        # Initialize posterior path
        initial_posterior_path = self._initialize_posterior_path(initial_posterior_path)
        self.fx_mus = tf.cast(tf.reshape(initial_posterior_path.mu, (self.time_grid.shape[0], self.state_dim)),
                              dtype=self._observations.dtype)
        self.fx_covs = tf.cast(
            tf.reshape(initial_posterior_path.cov, (self.time_grid.shape[0], self.state_dim, self.state_dim)),
            dtype=self._observations.dtype)

        self.obs_sites_indices = tf.where(tf.equal(self.time_grid[..., None], self._observations_time_points))[:, 0][
            ..., None]

        assert self.obs_sites_indices.shape[0] == self._observations.shape[0]

    def _initialize_initial_state_prior(self, prior_initial_state: Gaussian):
        """Initialize the prior on the initial state."""
        if prior_initial_state is None:
            prior_initial_state = Gaussian(mu=tf.zeros((self.state_dim,), dtype=self._observations.dtype),
                                           cov=self.dist_p.initial_covariance.numpy() * tf.ones((self.state_dim,
                                                                                                 self.state_dim),
                                                                                                dtype=self._observations.dtype))

        return prior_initial_state

    def _initialize_posterior_path(self, initial_posterior_path: Gaussian):
        """Initialize posterior path."""
        if initial_posterior_path is None:
            initial_posterior_path = Gaussian(mu=tf.zeros((self.time_grid.shape[0], self.state_dim),
                                                          dtype=self._observations.dtype),
                                              cov=tf.eye(self.state_dim, batch_shape=[self.time_grid.shape[0]],
                                                         dtype=self._observations.dtype))

        return initial_posterior_path

    def init_girsanov_sites(self):
        self.girsanov_sites = BTDGaussian(nat1=Parameter(tf.zeros((self.time_grid.shape[0], self.state_dim),
                                                                  dtype=self._observations.dtype)),
                                          nat2_diag=Parameter(
                                              tf.ones((self.time_grid.shape[0], self.state_dim, self.state_dim),
                                                      dtype=self._observations.dtype) * -1e-10),
                                          nat2_subdiag=Parameter(
                                              tf.ones((self.time_grid.shape[0] - 1, self.state_dim, self.state_dim),
                                                      dtype=self._observations.dtype) * -1e-10)
                                          )

    @property
    def time_points(self) -> TensorType:
        """
        Return the time points of the observations.
        :return: A tensor with shape ``batch_shape + [grid_size]``.
        """
        return self.time_grid

    def full_sites(self):
        """
        Sum natural parameters: girsanov_sites + data_sites + line_prior_sites
        """
        lin_p_btd_sites = ssm_to_btd_nat(self.dist_p)

        data_sites_nat1 = tf.scatter_nd(self.obs_sites_indices, self.data_sites.nat1, self.girsanov_sites.nat1.shape)
        data_sites_nat2 = tf.scatter_nd(self.obs_sites_indices, self.data_sites.nat2,
                                        self.girsanov_sites.nat2.block_diagonal.shape)

        q_nat1 = lin_p_btd_sites.nat1 + self.girsanov_sites.nat1 + data_sites_nat1
        q_btd_nat = lin_p_btd_sites.nat2 + self.girsanov_sites.nat2 + data_sites_nat2

        return q_nat1, q_btd_nat

    @property
    def dist_q(self) -> StateSpaceModel:
        """
        Construct the :class:`~markovflow.state_space_model.StateSpaceModel` representation of
        the posterior process indexed at the time points.
        """
        q_nat1, q_btd_nat = self.full_sites()

        As, offsets, chol_initial_cov, chol_process_cov, initial_mean = naturals_to_ssm_params(q_nat1,
                                                                                               q_btd_nat.block_diagonal,
                                                                                               q_btd_nat.block_sub_diagonal)

        q_ssm = StateSpaceModel(initial_mean=initial_mean, state_transitions=As, state_offsets=offsets,
                                chol_process_covariances=chol_process_cov,
                                chol_initial_covariance=chol_initial_cov)

        return q_ssm

    def local_objective(self, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> TensorType:
        """
        Calculate local loss in CVI.
        :param Fmu: Means with shape ``[..., latent_dim]``.
        :param Fvar: Variances with shape ``[..., latent_dim]``.
        :param Y: Observations with shape ``[..., observation_dim]``.
        :return: A local objective with shape ``[...]``.
        """
        return self.likelihood.variational_expectations(Fmu, Fvar, Y)

    def local_objective_and_gradients(self, Fmu: TensorType, Fvar: TensorType) -> [TensorType, TensorType]:
        """
        Return the local objective and its gradients with regard to the expectation parameters.
        :param Fmu: Means :math:`μ` with shape ``[..., latent_dim]``.
        :param Fvar: Variances :math:`σ²` with shape ``[..., latent_dim]``.
        :return: A local objective and gradient with regard to :math:`[μ, σ² + μ²]`.
        """
        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(
                input_tensor=self.local_objective(Fmu, Fvar, self._observations)
            )
        grads = g.gradient(local_obj, [Fmu, Fvar])
        # turn into gradient wrt μ, σ² + μ²
        grads = gradient_transformation_mean_var_to_expectation((Fmu, Fvar), grads)

        return local_obj, grads

    def KL_q_p(self):
        """
        """
        ssm_q_marginals_mean = self.dist_q.marginal_means
        ssm_q_marginals_covar = self.dist_q.marginal_covariances
        ssm_q_process_covar = tf.square(self.dist_q.cholesky_process_covariances)
        ssm_p_process_covar = tf.square(self.dist_p.cholesky_process_covariances)

        kl_q_p = SSM_KL_along_Gaussian_path(self.dist_q_forward,
                                            self.dist_p_forward,
                                            ssm_q_process_covar,
                                            ssm_p_process_covar,
                                            ssm_q_marginals_mean,
                                            ssm_q_marginals_covar)

        kl_q_0 = tfp.distributions.Normal(loc=self.dist_q.initial_mean, scale=self.dist_q.cholesky_initial_covariance)
        kl_p_0 = tfp.distributions.Normal(
            loc=self.dist_p.initial_mean, scale=self.dist_p.cholesky_initial_covariance
        )
        kl_0 = tfp.distributions.kl_divergence(kl_q_0, kl_p_0)

        return kl_q_p + kl_0

    def dist_q_forward(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluates f_q(x_t) = A(t)x(t) + b(t) where the shape of x(t) is `[num_batch, num_transitions, state_dim]`.
        """
        A = self.dist_q.state_transitions
        b = self.dist_q.state_offsets

        n_batch = x.shape[0]

        A_n = tf.squeeze(
            tf.repeat(A[None, ...], n_batch, axis=0), -1
        )  # [num_batch, n_transitions, 1]
        b_n = tf.repeat(b[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, 1]

        return A_n * x + b_n

    def dist_p_forward(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluates f_p(x_t) = A(t)x(t) + b(t) where the shape of x(t) is `[num_batch, num_transitions, state_dim]`.
        """
        A_p = self.dist_p.state_transitions
        b_p = self.dist_p.state_offsets
        n_batch = x.shape[0]

        A_p_n = tf.squeeze(
            tf.repeat(A_p[None, ...], n_batch, axis=0), -1
        )  # [num_batch, n_transitions, 1]
        b_p_n = tf.repeat(b_p[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, 1]

        return A_p_n * x + b_p_n

    def grad_kl_wrt_exp_param(self):
        return SSM_KL_with_grads_wrt_exp_params(self.dist_q, self.dist_p)

    def update_girsanov_sites(self, lr: float):
        """
        Update Girsanov sites using the gradient of KL[q || p] wrt the expectation parameters.
        """

        _, grad_kl = self.grad_kl_wrt_exp_param()

        data_sites_nat1 = tf.scatter_nd(self.obs_sites_indices, self.data_sites.nat1, self.girsanov_sites.nat1.shape)
        data_sites_nat2 = tf.scatter_nd(self.obs_sites_indices, self.data_sites.nat2,
                                        self.girsanov_sites.nat2.block_diagonal.shape)

        new_girsanov_sites_nat1 = self.girsanov_sites.nat1 + lr * (data_sites_nat1 - grad_kl[0])
        new_girsanov_sites_nat2 = self.girsanov_sites.nat2.block_diagonal + lr * (data_sites_nat2 - grad_kl[1])
        new_girsanov_sites_nat3 = self.girsanov_sites.nat2.block_sub_diagonal - lr * grad_kl[2]

        self.girsanov_sites.nat1.assign(new_girsanov_sites_nat1)
        self.girsanov_sites.nat2.block_diagonal.assign(new_girsanov_sites_nat2)
        self.girsanov_sites.nat2.block_sub_diagonal.assign(new_girsanov_sites_nat3)

        # Update the current posterior
        self.fx_mus, self.fx_covs = self.dist_q.marginals

    def update_data_sites(self, lr: float):
        """
        Update of the data-sites using the gradient of KL[q || p] wrt the expectation parameters.
        """
        fx_mus = tf.gather_nd(tf.reshape(self.fx_mus, (-1, self.state_dim)), self.obs_sites_indices)
        fx_covs = tf.gather_nd(tf.reshape(self.fx_covs, (-1, self.state_dim, self.state_dim)), self.obs_sites_indices)

        # get gradient of variational expectations wrt the expectation parameters μ, σ² + μ²
        _, grads_ve = self.local_objective_and_gradients(fx_mus, fx_covs)
        new_data_nat1 = (1 - lr) * self.data_sites.nat1 + lr * grads_ve[0]
        new_data_nat2 = (1 - lr) * self.data_sites.nat2 + lr * grads_ve[1]

        self.data_sites.nat1.assign(new_data_nat1)
        self.data_sites.nat2.assign(new_data_nat2)

        # Update the current posterior
        self.fx_mus, self.fx_covs = self.dist_q.marginals

    def variational_expectation(self, fx_mus: TensorType = None, fx_covs: TensorType = None) -> TensorType:
        """
        Expected log-likelihood under the current variational posterior
        """

        if fx_mus is None or fx_covs is None:
            fx_mus, fx_covs = self.dist_q.marginals

        fx_mus_obs = tf.gather_nd(tf.reshape(fx_mus, (-1, self.state_dim)), self.obs_sites_indices)
        fx_covs_obs = tf.gather_nd(tf.reshape(fx_covs, (-1, self.state_dim, self.state_dim)), self.obs_sites_indices)
        # fx_covs_obs = tf.linalg.diag_part(fx_covs_obs)

        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self.likelihood.variational_expectations(
                fx_mus_obs, fx_covs_obs, self._observations
            )
        )
        return ve_fx

    def classic_elbo(self) -> TensorType:
        """
        Compute the ELBO.
        ELBO of the model is defined as:
            E_{q}[log p(Y|X)] - KL[q||p]]
        """
        # s ~ q(s) = N(μ, P)
        dist_q = self.dist_q
        fx_mus, fx_covs = dist_q.marginals

        ve_fx = self.variational_expectation(fx_mus, fx_covs)
        kl_fx = self.KL_q_p()

        return ve_fx - kl_fx

    def loss(self) -> tf.Tensor:
        pass

    @property
    def posterior(self) -> PosteriorProcess:
        pass

    def grad_KL_wrt_prior_params(self):
        """
        Calculate the gradient of the KL term wrt the prior parameters
        """
        pass


class CVISitesSDE(CVISitesSSM, ABC):
    """
    Provides a site-based parameterization to the variational posterior over the state trajectory of a SDE.
    ELBO of the model is defined as:
                    E_{q}[log p(Y|X)] - KL[q||p]
    """

    def __init__(
            self,
            prior_sde: SDE,
            time_grid: TensorType,
            input_data: Tuple[TensorType, TensorType],
            likelihood: Likelihood,
            prior_initial_state: Gaussian = None,
            initial_posterior_path: Gaussian = None,
            stabilize_ssm: bool = True,
            clip_state_transitions: Tuple = (-1., 1.)
    ) -> None:
        """
        :param prior_sde: Prior SDE over the latent states, x.
        :param time_grid: Grid over time with shape ``batch_shape + [grid_size]``
        :param input_data: A tuple containing the observed data:
            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``
        :param likelihood: A likelihood for the observations of the model.
        :param prior_initial_state: A Gaussian prior on the initial state.
        :param initial_posterior_path: A Gaussian prior on the initial posterior path.

        Note: Currently, batching isn't supported!
        """
        self.state_dim = input_data[-1].shape[-1]
        self.prior_sde = prior_sde
        self.prior_initial_state = self._initialize_initial_state_prior(prior_initial_state)
        super().__init__(None, time_grid, input_data, likelihood,
                         initial_posterior_path=initial_posterior_path, prior_initial_state=self.prior_initial_state)
        self.dist_p_linearized = None
        self.stabilize_ssm = stabilize_ssm
        self.clip_state_transitions = clip_state_transitions
        self.set_linearized_prior()

    def set_linearized_prior(self):
        """
        Linearize the SDE on the current posterior
        """
        posterior_path = Gaussian(mu=self.fx_mus[1:], cov=self.fx_covs[1:])
        dist_p_linearized = linearize_sde(self.prior_sde, transition_times=self.time_grid,
                                          initial_state=self.prior_initial_state, linearization_path=posterior_path)
        self.dist_p_linearized = remove_batch_from_ssm(dist_p_linearized)

        if self.stabilize_ssm:
            # Get a stable SSM
            clip_state_transitions = tf.clip_by_value(self.dist_p_linearized.state_transitions,
                                                      self.clip_state_transitions[0], self.clip_state_transitions[1])
            clip_state_offsets = tf.clip_by_value(self.dist_p_linearized.state_offsets,
                                                  self.clip_state_transitions[0], self.clip_state_transitions[1])
            stable_ssm = StateSpaceModel(
                state_transitions=clip_state_transitions,
                state_offsets=clip_state_offsets,
                chol_process_covariances=self.dist_p_linearized.cholesky_process_covariances,
                initial_mean=self.dist_p_linearized.initial_mean,
                chol_initial_covariance=self.dist_p_linearized.cholesky_initial_covariance
            )
            self.dist_p = stable_ssm
        else:
            self.dist_p = self.dist_p_linearized

    def _initialize_initial_state_prior(self, prior_initial_state: Gaussian):
        """
        Initialize prior on initial state.
        """
        if prior_initial_state is None:
            prior_initial_state = Gaussian(mu=tf.zeros((self.state_dim,), dtype=self.prior_sde.q.dtype),
                                           cov=self.prior_sde.q.numpy() * tf.ones((self.state_dim,
                                                                                   self.state_dim),
                                                                                  dtype=self.prior_sde.q.dtype))

        return prior_initial_state

    def KL_q_p(self):
        """
        KL between approximating posterior and true prior i.e. the SDE.
        """
        ssm_q_marginals_mean = self.dist_q.marginal_means
        ssm_q_marginals_covar = self.dist_q.marginal_covariances
        ssm_q_process_covar = tf.square(self.dist_q.cholesky_process_covariances)
        ssm_p_process_covar = self.dt * tf.repeat(self.prior_sde.q[None, ...], ssm_q_process_covar.shape[0], axis=0)

        def dist_q_forward(x: tf.Tensor) -> tf.Tensor:
            """
            Evaluates f_q(x_t) = A(t)x(t) + b(t) where the shape of x(t) is `[num_batch, num_transitions, state_dim]`.
            """
            n_batch = x.shape[0]

            A_n = tf.repeat(self.dist_q.state_transitions[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, D]
            b_n = tf.repeat(self.dist_q.state_offsets[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, 1]

            val = tf.stop_gradient((A_n @ x[..., None])[:, :, :, 0] + b_n)
            return val

        def prior_sde_ssm_forward(x, t=None):
            return x + self.dt * self.prior_sde.drift(x, t)

        kl_q_p = SSM_KL_along_Gaussian_path(dist_q_forward,
                                            prior_sde_ssm_forward,
                                            ssm_q_process_covar,
                                            ssm_p_process_covar,
                                            ssm_q_marginals_mean,
                                            ssm_q_marginals_covar,
                                            )

        kl_q_0 = tfp.distributions.MultivariateNormalFullCovariance(loc=self.dist_q.initial_mean,
                                                        covariance_matrix=self.dist_q.cholesky_initial_covariance @ tf.transpose(self.dist_q.cholesky_initial_covariance))
        kl_p_0 = tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.dist_p.initial_mean, covariance_matrix=self.dist_p.cholesky_initial_covariance @
                                                            tf.transpose(self.dist_p.cholesky_initial_covariance))

        kl_0 = tfp.distributions.kl_divergence(kl_q_0, kl_p_0)

        return kl_q_p + kl_0

    def grad_kl_wrt_exp_param(self):
        """
        Gradeint of KL[q||p] wrt the expectation parameters of q.
        """
        return SDE_SSM_KL_with_grads_wrt_exp_params(self.dist_q, self.prior_sde, self.dt, self.prior_initial_state,
                                                    self.time_grid)

    def grad_KL_wrt_prior_params(self):
        """
        Calculates the gradient of the KL term wrt the prior parameters.
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch(self.prior_sde.trainable_variables)
            KL_val = self.KL_q_p()

        grads = g.gradient(KL_val, self.prior_sde.trainable_variables)
        return grads

    def grad_VE_wrt_prior_params(self):
        """
        Calculates the gradient of the VE term wrt the prior parameters.
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch(self.prior_sde.trainable_variables)
            self.set_linearized_prior()
            fx_mus, fx_covs = self.dist_q.marginals

            ve_val = -1.0 * self.variational_expectation(fx_mus, fx_covs)

        grads = g.gradient(ve_val, self.prior_sde.trainable_variables)
        return grads
