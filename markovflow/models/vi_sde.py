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
    Module containing a model for variational inference in SDE.

    @inproceedings{pmlr-v1-archambeau07a,
      title = 	 {Gaussian Process Approximations of Stochastic Differential Equations},
      author = 	 {Archambeau, Cedric and Cornford, Dan and Opper, Manfred and Shawe-Taylor, John},
      booktitle = 	 {Gaussian Processes in Practice},
      pages = 	 {1--16},
      year = 	 {2007},
      editor = 	 {Lawrence, Neil D. and Schwaighofer, Anton and Quiñonero Candela, Joaquin},
      volume = 	 {1},
      series = 	 {Proceedings of Machine Learning Research},
      address = 	 {Bletchley Park, UK},
      month = 	 {12--13 Jun},
      publisher =    {PMLR},
      pdf = 	 {http://proceedings.mlr.press/v1/archambeau07a/archambeau07a.pdf},
      url = 	 {https://proceedings.mlr.press/v1/archambeau07a.html},
    }


    @inproceedings{NIPS2007_818f4654,
         author = {Archambeau, C\'{e}dric and Opper, Manfred and Shen, Yuan and Cornford, Dan and Shawe-taylor, John},
         booktitle = {Advances in Neural Information Processing Systems},
         editor = {J. Platt and D. Koller and Y. Singer and S. Roweis},
         pages = {},
         publisher = {Curran Associates, Inc.},
         title = {Variational Inference for Diffusion Processes},
         url = {https://proceedings.neurips.cc/paper/2007/file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf},
         volume = {20},
         year = {2007}
    }
"""

import tensorflow as tf
from gpflow.likelihoods import Likelihood
from gpflow.probability_distributions import Gaussian
from tensorflow_probability import distributions
from markovflow.state_space_model import StateSpaceModel

from markovflow.sde import SDE
from markovflow.sde.drift import LinearDrift
from markovflow.sde.sde_utils import squared_drift_difference_along_Gaussian_path

CLIP_MIN = -5000
CLIP_MAX = 5000


class VariationalMarkovGP:
    """
        Variational approximation to a non-linear SDE by a time-varying Gauss-Markov Process of the form
        dx(t) = -A(t) dt + b(t) dW(t)
    """

    def __init__(self, input_data: [tf.Tensor, tf.Tensor], prior_sde: SDE, grid: tf.Tensor, likelihood: Likelihood,
                 prior_initial_state: distributions.Normal = None, stabilize_system: bool = False):
        """
        Initialize the model.
        """
        # assert prior_sde.state_dim == 1, "Currently only 1D is supported."

        self._time_points, self.observations = input_data
        self.prior_sde = prior_sde
        self.likelihood = likelihood
        self.state_dim = self.prior_sde.state_dim
        self.DTYPE = self.observations.dtype

        self.grid = grid
        self.num_states = grid.shape[0]
        self.num_transitions = self.num_states - 1

        self.dt = float(self.grid[1] - self.grid[0])
        self.observations_time_points = self._time_points

        self.A = tf.zeros((self.num_transitions, self.state_dim, self.state_dim), dtype=self.DTYPE)
        self.b = tf.zeros((self.num_transitions, self.state_dim), dtype=self.DTYPE)

        # p(x0)
        if prior_initial_state:
            self.prior_initial_state = prior_initial_state
        else:
            self.prior_initial_state = distributions.MultivariateNormalFullCovariance(
                loc=tf.zeros((self.state_dim,), dtype=self.DTYPE),
                covariance_matrix=self.prior_sde.q * tf.eye(self.state_dim, dtype=self.DTYPE))

        # q(x0)
        self.q_initial_state = distributions.MultivariateNormalTriL(loc=tf.identity(self.prior_initial_state.loc),
                                                                    scale_tril=tf.identity(
                                                                        self.prior_initial_state.scale_tril))

        self.lambda_lagrange = tf.zeros((self.num_transitions, self.state_dim), dtype=self.DTYPE)
        self.psi_lagrange = tf.eye(self.state_dim, batch_shape=(self.num_transitions,), dtype=self.DTYPE) * 1e-10

        self.dist_q_ssm = None
        self.stabilize_system = stabilize_system

    def kf_forward_pass(self, ssm: StateSpaceModel):
        P_0 = tf.matmul(
            ssm.cholesky_initial_covariance,
            ssm.cholesky_initial_covariance,
            transpose_b=True,
        )
        mu_0 = ssm.initial_mean
        Q_s = tf.matmul(
            ssm.cholesky_process_covariances,
            ssm.cholesky_process_covariances,
            transpose_b=True,
        )
        A_s = ssm.state_transitions
        b_s = ssm.state_offsets

        def step(pred_mus, pred_covs, counter):
            A_k = A_s[..., counter, :, :]  # [... state_dim, state_dim]
            b_k = b_s[..., counter, :]  # [...  1, state_dim]
            Q_k = Q_s[..., counter, :, :]  # [... state_dim, state_dim]
            filter_cov = pred_covs[..., -1, :, :]
            filter_mean = pred_mus[..., -1, :]

            # propagate
            pred_mean = tf.linalg.matvec(A_k, filter_mean) + b_k
            pred_cov = A_k @ tf.matmul(filter_cov, A_k, transpose_b=True) + Q_k

            # stick the new mean and covariance to their accumulators and increment the counter
            return (
                tf.concat([pred_mus, pred_mean[..., None, :]], axis=-2),
                tf.concat([pred_covs, pred_cov[..., None, :, :]], axis=-3),
                counter + 1,
            )

        # set up the loop variables and shape invariants
        # [... 1, state_dim] and [... 1, state_dim, state_dim]
        loop_vars = (
            mu_0[..., None, :],
            P_0[..., None, :, :],
            tf.constant(0, tf.int32),
        )

        batch_shape = ssm.batch_shape
        state_dim = ssm.state_dim
        num_transitions = ssm.num_transitions
        shape_invars = (
            tf.TensorShape(batch_shape + (None, state_dim)),
            tf.TensorShape(batch_shape + (None, state_dim, state_dim)),
            tf.TensorShape([]),
        )

        def cond(____, _____, counter):
            return counter < num_transitions

        pred_mus, pred_covs, _ = tf.while_loop(
            cond=cond, body=step, loop_vars=loop_vars, shape_invariants=shape_invars
        )

        return pred_mus, pred_covs

    @property
    def forward_pass(self) -> (tf.Tensor, tf.Tensor):
        """
        Computes the mean and variance of the SDE using SSM.

        The returned m and S have initial values appended too.
        """
        # [num_transitions, state_dim, state_dim]
        q = tf.repeat(tf.reshape(self.prior_sde.q, (1, self.state_dim, self.state_dim)), self.A.shape[0], axis=0)

        self.dist_q_ssm = LinearDrift(A=-1 * self.A, b=self.b).to_ssm(q=q, initial_mean=self.q_initial_state.loc,
                                                                      transition_times=self.grid,
                                                                      initial_chol_covariance=self.q_initial_state.scale_tril)

        if self.stabilize_system:
            # Get a stable SSM
            clip_state_transitions = tf.where(tf.math.is_nan(self.dist_q_ssm.state_transitions), 1e-8,
                                              self.dist_q_ssm.state_transitions)
            clip_state_offsets = tf.where(tf.math.is_nan(self.dist_q_ssm.state_offsets), 1e-8,
                                          self.dist_q_ssm.state_offsets)

            clip_state_transitions = tf.clip_by_value(clip_state_transitions, -1., 1.)
            clip_state_offsets = tf.clip_by_value(clip_state_offsets, -1., 1.)

            self.dist_q_ssm = StateSpaceModel(
                state_transitions=clip_state_transitions,
                state_offsets=clip_state_offsets,
                chol_process_covariances=self.dist_q_ssm.cholesky_process_covariances,
                initial_mean=self.dist_q_ssm.initial_mean,
                chol_initial_covariance=self.dist_q_ssm.cholesky_initial_covariance
            )

        marginal_means, marginal_covs = self.dist_q_ssm.marginals  # self.kf_forward_pass(self.dist_q_ssm)

        return marginal_means[0], marginal_covs[0]  # 0 to remove batching

    def _grad_E_sde(self, m: tf.Tensor, S: tf.Tensor):
        """
        Gradient of E_sde wrt m and S.
        """
        m = m[:-1]
        S = S[:-1]

        tf.debugging.assert_shapes(
            [
                (m, ("num_transitions", "state_dim")),
                (S, ("num_transitions", "state_dim", "state_dim")),
                (self.b, ("num_transitions", "state_dim")),
                (self.A, ("num_transitions", "state_dim", "state_dim"))
            ]
        )

        with tf.GradientTape(persistent=True) as g:
            g.watch([m, S])
            E_sde = self.E_sde(m, S)

        dE_dm, dE_dS = g.gradient(E_sde, [m, S])

        # As E_sde is approximated with Riemann sum, we remove the contribution of dt
        dE_dm = dE_dm / self.dt
        dE_dS = dE_dS / self.dt

        tf.debugging.assert_shapes(
            [
                (m, ("num_transitions", "state_dim")),
                (dE_dm, ("num_transitions", "state_dim")),
                (dE_dS, ("num_transitions", "state_dim", "state_dim"))
            ]
        )

        return dE_dm, dE_dS

    def update_initial_statistics(self, lr: float):
        """
        Update the initial statistics.
        """
        q_initial_mean = self.prior_initial_state.loc - ((
                    self.prior_initial_state.scale_tril @ tf.transpose(self.prior_initial_state.scale_tril)) @\
                         self.lambda_lagrange[0][..., None])[..., 0]

        prior_chol_cov_inv = tf.linalg.cholesky_solve(self.prior_initial_state.scale_tril,
                                                      tf.eye(self.state_dim, dtype=self.DTYPE))
        prior_cov_inv = prior_chol_cov_inv @ tf.transpose(prior_chol_cov_inv)
        q_initial_cov_inv = prior_cov_inv + 2 * self.psi_lagrange[0]
        q_initial_cov = tf.linalg.inv(q_initial_cov_inv)

        q_initial_mean = (1 - lr) * self.q_initial_state.loc + lr * q_initial_mean
        q_initial_cov = (1 - lr) * (self.q_initial_state.scale_tril @ tf.transpose(self.q_initial_state.scale_tril)) + lr * q_initial_cov

        self.q_initial_state = distributions.MultivariateNormalTriL(loc=q_initial_mean,
                                                                    scale_tril=tf.linalg.cholesky(q_initial_cov))

    def _jump_conditions(self, m: tf.Tensor, S: tf.Tensor):
        """
        Declare jump conditions on a bigger grid with values only where the observations are available.
        """

        tf.debugging.assert_shapes(
            [
                (self.A, ("num_transtions", "state_dim", "state_dim")),
                (m, ("num_transtions + 1", "state_dim")),
                (S, ("num_transtions + 1", "state_dim", "state_dim")),
            ]
        )

        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0]
        m_obs_t = tf.gather(m, indices, axis=0)
        S_obs_t = tf.gather(S, indices, axis=0)

        _, grads = self.local_objective_and_gradients(m_obs_t, S_obs_t)

        # Put grads in a bigger grid back
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0][..., None]

        d_obs_m = tf.scatter_nd(indices, grads[0], m.shape)
        d_obs_S = tf.scatter_nd(indices, grads[1], S.shape)

        return d_obs_m, d_obs_S

    def update_lagrange(self, m: tf.Tensor, S: tf.Tensor):
        """
        Backward pass incorporating jump conditions.
        This function updates the lagrange multiplier values.

        d psi(t)/dt = 2 * psi(t) * A(t) - dE_{sde}/ds(t)
        d lambda(t)/dt = A(t).T * lambda(t) - dE_{sde}/dm(t)

        At the time of observation, jump conditions:
            psi(t) = psi(t) + dE_{obs}/dS(t)
            lambda(t) = lambda(t) + dE_{obs}/dm(t)

        """

        tf.debugging.assert_shapes(
            [
                (self.A, ("num_transitions", "state_dim", "state_dim")),
                (m, ("num_transitions + 1", "state_dim")),
                (S, ("num_transitions + 1", "state_dim", "state_dim")),
            ]
        )

        dEdm, dEdS = self._grad_E_sde(m, S)
        d_obs_m, d_obs_S = self._jump_conditions(m, S)

        if self.stabilize_system:
            dEdm = tf.where(tf.math.is_nan(dEdm), 1e-8, dEdm)
            dEdS = tf.where(tf.math.is_nan(dEdS), 1e-8, dEdS)
            d_obs_m = tf.where(tf.math.is_nan(d_obs_m), 1e-8, d_obs_m)
            d_obs_S = tf.where(tf.math.is_nan(d_obs_S), 1e-8, d_obs_S)

            dEdm = tf.clip_by_value(dEdm, CLIP_MIN, CLIP_MAX)
            dEdS = tf.clip_by_value(dEdS, CLIP_MIN, CLIP_MAX)
            d_obs_m = tf.clip_by_value(d_obs_m, CLIP_MIN, CLIP_MAX)
            d_obs_S = tf.clip_by_value(d_obs_S, CLIP_MIN, CLIP_MAX)

        tf.debugging.assert_shapes(
            [
                (dEdm, ("num_transitions", "state_dim")),
                (dEdS, ("num_transitions", "state_dim", "state_dim")),
                (d_obs_m, ("num_transitions + 1", "state_dim")),
                (d_obs_S, ("num_transitions + 1", "state_dim", "state_dim")),
            ]
        )

        self.lambda_lagrange = tf.zeros_like(self.lambda_lagrange)
        self.psi_lagrange = tf.eye(self.state_dim, batch_shape=(self.num_transitions,), dtype=self.DTYPE) * 1e-10

        for t in range(self.num_transitions - 1, 0, -1):
            d_psi = self.psi_lagrange[t] @ self.A[t] + self.psi_lagrange[t] @ self.A[t] - dEdS[t]
            d_lambda = (self.A[t] @ self.lambda_lagrange[t][..., None])[..., 0] - dEdm[t]

            psi_lagrange_val = self.psi_lagrange[t] - self.dt * d_psi - d_obs_S[t]
            lambda_lagrange_val = self.lambda_lagrange[t] - self.dt * d_lambda - d_obs_m[t]

            self.psi_lagrange = tf.tensor_scatter_nd_update(self.psi_lagrange, [[t - 1]],
                                                            psi_lagrange_val[None, ...])
            self.lambda_lagrange = tf.tensor_scatter_nd_update(self.lambda_lagrange, [[t - 1]],
                                                               lambda_lagrange_val[None, ...])

    def local_objective(self, Fmu: tf.Tensor, Fvar: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Calculate local loss..

        :param Fmu: Means with shape ``[..., latent_dim]``.
        :param Fvar: Variances with shape ``[..., latent_dim]``.
        :param Y: Observations with shape ``[..., observation_dim]``.
        :return: A local objective with shape ``[...]``.
        """
        return self.likelihood.variational_expectations(Fmu, Fvar, Y)

    def local_objective_and_gradients(self, Fmu: tf.Tensor, Fvar: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Return the local objective and its gradients.

        :param Fmu: Means :math:`μ` with shape ``[..., latent_dim]``.
        :param Fvar: Variances :math:`σ²` with shape ``[..., latent_dim]``.
        :return: A local objective and gradient with regard to :math:`[μ, σ²]`.
        """
        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(
                input_tensor=self.local_objective(Fmu, Fvar, self.observations)
            )
        grads = g.gradient(local_obj, [Fmu, Fvar])

        return local_obj, grads

    def update_param(self, m: tf.Tensor, S: tf.Tensor, lr: float):
        """
        Update the params A(t) and b(t).

        \tilde{A(t)} = - <\frac{df}{dx}>_{qt} + 2 Q \psi(t)
        \tilde{b(t)} = <f(x)>_{qt} + \tilde{A(t)} * m(t) - Q \lambda(t)

        A(t) = (1 - lr) * A(t) + lr * \tilde{A(t)}
        b(t) = (1 - lr) * b(t) + lr *  \tilde{b(t)}

        """

        # Remove the last state
        m = m[:-1]
        S = S[:-1]

        if self.stabilize_system:
            self.psi_lagrange = tf.where(tf.math.is_nan(self.psi_lagrange), 1e-8, self.psi_lagrange)
            self.lambda_lagrange = tf.where(tf.math.is_nan(self.lambda_lagrange), 1e-8, self.lambda_lagrange)
            self.psi_lagrange = tf.clip_by_value(self.psi_lagrange, CLIP_MIN, CLIP_MAX)
            self.lambda_lagrange = tf.clip_by_value(self.lambda_lagrange, CLIP_MIN, CLIP_MAX)

        var = self.prior_sde.q

        expected_gradient_drift = tf.squeeze(-self.prior_sde.expected_gradient_drift(m[None, ...], S), axis=0)
        expected_drift = tf.squeeze(self.prior_sde.expected_drift(m[None, ...], S), axis=0)

        if self.state_dim == 1:
            expected_gradient_drift = expected_gradient_drift[..., None]

        A_tilde = expected_gradient_drift + 2. * tf.repeat(var[None, ...], m.shape[0], axis=0) @ self.psi_lagrange
        b_tilde = expected_drift + (A_tilde @ m[..., None])[..., 0] - (tf.repeat(var[None, ...], m.shape[0], axis=0) @ self.lambda_lagrange[..., None])[..., 0]

        assert A_tilde.shape == self.A.shape
        assert b_tilde.shape == self.b.shape

        self.A = (1 - lr) * self.A + lr * A_tilde
        self.b = (1 - lr) * self.b + lr * b_tilde

    def KL_initial_state(self):
        """
        KL[q(x0) || p(x0)]
        """
        return distributions.kl_divergence(self.q_initial_state, self.prior_initial_state)

    def E_sde(self, m: tf.Tensor = None, S: tf.Tensor = None):
        """
        Calculate E_{sde} term.
        """
        if m is None or S is None:
            m, S = self.forward_pass
            # remove the final state
            m = m[:-1]
            S = S[:-1]

        E_sde = squared_drift_difference_along_Gaussian_path(self.prior_sde, LinearDrift(-1 * self.A, self.b),
                                                             Gaussian(m, S), self.dt)
        return E_sde

    def elbo(self, m=None, S=None) -> float:
        """ Variational lower bound to the marginal likelihood """
        if m is None:
            m, S = self.forward_pass

        # KL
        E_sde = self.E_sde()
        KL_q0_p0 = self.KL_initial_state()

        # E_obs
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0]
        m_obs_t = tf.gather(m, indices, axis=0)
        S_obs_t = tf.gather(S, indices, axis=0)

        E_obs = self.likelihood.variational_expectations(m_obs_t, S_obs_t, self.observations)
        E_obs = tf.reduce_sum(E_obs)

        E = E_obs - E_sde - KL_q0_p0

        return E.numpy().item()

    def grad_prior_sde_params(self):
        """
        Get the gradient of E_sde wrt prior sde params.
        """
        with tf.GradientTape(persistent=True) as g:
            m, S = self.forward_pass
            m = tf.stop_gradient(m[1:])
            S = tf.stop_gradient(S[1:])
            g.watch(self.prior_sde.trainable_variables)
            E_sde = self.E_sde(m, S)

        grads = g.gradient(E_sde, self.prior_sde.trainable_variables)

        return grads

    def grad_initial_state(self):
        """
        Get the gradient of KL[initial_state] wrt prior_x0 params
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch([self.prior_initial_state.loc, self.prior_initial_state.scale])
            val = self.KL_initial_state()

        grads = g.gradient(val, [self.prior_initial_state.loc, self.prior_initial_state.scale])

        return grads
