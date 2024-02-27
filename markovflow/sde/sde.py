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

from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf
from gpflow.quadrature import mvnquad


class SDE(tf.Module, ABC):
    """
    Abstract class representing Stochastic Differential Equation.

    ..math::
     &dx(t)/dt = f(x(t),t) + l(x(t),t) w(t)

    """

    def __init__(self, state_dim=1):
        """
        :param state_dim: The output dimension of the kernel.
        """
        super().__init__()
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the sde.
        """
        return self._state_dim

    @abstractmethod
    def drift(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Drift function of the SDE i.e. `f(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, state_dim)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """
        Diffusion function of the SDE i.e. `l(x(t),t)`

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError

    def gradient_drift(self, x: tf.Tensor, t: tf.Tensor = tf.zeros((1, 1))) -> tf.Tensor:
        """
        Calculates the gradient of the drift wrt the states x(t).

        ..math:: df(x(t))/dx(t)

        :param x: states with shape (num_states, state_dim).
        :param t: time of states with shape (num_states, 1), defaults to zero.

        :return: the gradient of the SDE drift with shape (num_states, state_dim).
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            drift_val = self.drift(x, t)
            dfx = tape.gradient(drift_val, x)
        return dfx

    def expected_drift(self, q_mean: tf.Tensor, q_covar: tf.Tensor) -> tf.Tensor:
        """
        Calculates the Expectation of the drift under the provided Gaussian over states.

        ..math:: E_q(x(t))[f(x(t))]

        :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
        :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

        :return: the expectation value with shape (n_batch, num_states, state_dim).
        """
        fx = lambda x: self.drift(x=x, t=tf.zeros(x.shape[0], 1))

        n_batch, n_states, state_dim = q_mean.shape
        q_mean = tf.reshape(q_mean, (-1, state_dim))
        q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))

        val = mvnquad(fx, q_mean, q_covar, H=10, Din=state_dim, Dout=(state_dim,))

        val = tf.reshape(val, (n_batch, n_states, state_dim))
        return val

    def expected_gradient_drift(self, q_mean: tf.Tensor, q_covar: tf.Tensor) -> tf.Tensor:
        """
         Calculates the Expectation of the gradient of the drift under the provided Gaussian over states

        ..math:: E_q(.)[f'(x(t))]

        :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
        :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

        :return: the expectation value with shape (n_batch, num_states, state_dim).
        """
        n_batch, n_states, state_dim = q_mean.shape
        q_mean = tf.reshape(q_mean, (-1, state_dim))
        q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))
        val = mvnquad(self.gradient_drift, q_mean, q_covar, H=10, Din=state_dim, Dout=(state_dim,))

        val = tf.reshape(val, (n_batch, n_states, state_dim))
        return val


class OrnsteinUhlenbeckSDE(SDE, ABC):
    """
    Ornstein-Uhlenbeck SDE represented by

    ..math:: dx(t) = -λ x(t) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, decay: float = 1., q: tf.Tensor = tf.ones((1, 1)), trainable: bool = False):
        """
        Initialize the Ornstein-Uhlenbeck SDE.

        :param decay: λ, a float value.
        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(OrnsteinUhlenbeckSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.decay = tf.Variable(initial_value=decay, trainable=trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the Ornstein-Uhlenbeck process
        ..math:: f(x(t), t) = -λ x(t)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return -self.decay * x

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the Ornstein-Uhlenbeck process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class DoubleWellSDE(SDE, ABC):
    """
    Double-Well SDE represented by

    ..math:: dx(t) = f(x(t)) dt + dB(t),

    where f(x(t)) = 4 x(t) (1 - x(t)^2) and the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, q: tf.Tensor = tf.ones((1, 1)), scale_trainable: bool = False, c_trainable: bool = False,
                 scale: float = 4., c: float = 1.):
        """
        Initialize the Double-Well SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(DoubleWellSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.scale = tf.Variable(initial_value=scale, trainable=scale_trainable, dtype=self.q.dtype)
        self.c = tf.Variable(initial_value=c, trainable=c_trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the double-well process
        ..math:: f(x(t), t) = 4 x(t) (1 - x(t)^2)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return self.scale * x * (self.c - tf.square(x))

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the double-well process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class BenesSDE(SDE, ABC):
    """
    Benes SDE
    ..math:: dx(t) = \theta tanh(x(t)) dt + dB(t),

    where the spectral density of the Brownian motion is specified by q.
    """
    def __init__(self, theta: float = 1, q: tf.Tensor = tf.ones((1, 1)), trainable=False):
        """
        Initialize the Benes SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(BenesSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.theta = tf.Variable(initial_value=theta, trainable=trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the double-well process
        ..math:: f(x(t), t) =tanh(x(t))

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return self.theta * tf.math.tanh(x)

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the double-well process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class SineDiffusionSDE(SDE, ABC):
    """
    Sine diffusion SDE represented by

    ..math:: dx(t) = sin(x(t) - \theta) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, theta: float = 0, q: tf.Tensor = tf.ones((1, 1)), trainable=False):
        """
        Initialize the SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(SineDiffusionSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.theta = tf.Variable(initial_value=theta, trainable=trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the process
        ..math:: f(x(t), t) = sin(x(t))

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.math.sin(x - self.theta)

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the Ornstein-Uhlenbeck process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class SqrtDiffusionSDE(SDE, ABC):
    """
    Sqrt diffusion SDE represented by

    ..math:: dx(t) = sqrt(theta |x(t)|) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, theta: float = 1, q: tf.Tensor = tf.ones((1, 1)), trainable=False):
        """
        Initialize the SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(SqrtDiffusionSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.theta = tf.Variable(initial_value=theta, trainable=trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the process
        ..math:: f(x(t), t) = sqrt(|x(t)|)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.math.sqrt(self.theta * tf.math.abs(x))

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class MLPDrift(SDE, ABC):
    """
    MLP diffusion SDE represented by

    ..math:: dx(t) = f(x(t)) dt + dB(t), the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, q: tf.Tensor = tf.ones((1, 1))):
        """
        Initialize the SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(MLPDrift, self).__init__(state_dim=q.shape[0])
        self.MLP = tf.keras.Sequential([
                        tf.keras.layers.Dense(3, activation='relu',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1)),
                        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1))
                    ])
        self.q = q

    def get_batches(self, x, batch_size=10000):
        """
        Get batches from x
        """
        i = 0
        batches = []
        while (i + batch_size) < x.shape[0]:
            batches.append(x[i: i+batch_size])
            i = i + batch_size

        batches.append(x[i:])

        return batches

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the process

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        input_shape = x.shape
        x = tf.reshape(x, (-1, 1))
        drift_val = None
        if x.shape[0] > 10000:
            x_batches = self.get_batches(x)
            for x in x_batches:
                if drift_val is None:
                    drift_val = tf.cast(self.MLP(x), x.dtype)
                else:
                    drift_val = tf.concat([drift_val, tf.cast(self.MLP(x), x.dtype)], axis=0)
        else:
            drift_val = tf.cast(self.MLP(x), x.dtype)
        return tf.reshape(drift_val, input_shape)

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)


class VanderPolOscillatorSDE(SDE, ABC):
    """
    Van der Pol's Oscillator SDE represented by

    ..math:: dx(t) = f(x(t), t) + dB(t), the spectral density of the Brownian motion is specified by q.

    f_1(x(t), t) = \theta * (x_1 - 1/3 * x_1^3 - x_2)
    f_1(x(t), t) = (1/ \theta) * x_1

    """

    def __init__(self, a: float = 1., tau: float = 1., q: tf.Tensor = tf.ones((1, 1)), trainable=False):
        """
        Initialize the SDE.

        :param q: spectral density of the Brownian motion ``(state_dim, state_dim)``.
        """
        super(VanderPolOscillatorSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        self.a = tf.Variable(initial_value=a, trainable=trainable, dtype=self.q.dtype)
        self.tau = tf.Variable(initial_value=tau, trainable=trainable, dtype=self.q.dtype)

    def drift(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Drift of the process
        ..math:: f(x(t), t) = sqrt(|x(t)|)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Drift value i.e. `f(x(t), t)` with shape ``(n_batch, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        dx1 = self.a * (x[..., 0] - (1 / 3) * tf.math.pow(x[..., 0], 3) - x[..., 1])
        dx2 = (1 / self.a) * x[..., 0]

        dx = self.tau * tf.stack([dx1, dx2], axis=-1)
        return dx

    def diffusion(self, x: tf.Tensor, t: tf.Tensor = None) -> tf.Tensor:
        """
        Diffusion of the process
        ..math:: l(x(t), t) = sqrt(q)

        :param x: state at `t` i.e. `x(t)` with shape ``(n_batch, state_dim)``.
        :param t: time `t` with shape ``(n_batch, 1)``.

        :return: Diffusion value i.e. `l(x(t), t)` with shape ``(n_batch, state_dim, state_dim)``.
        """
        assert x.shape[-1] == self.state_dim
        return tf.ones_like(x[..., None]) * tf.linalg.cholesky(self.q)

    def gradient_drift(self, x: tf.Tensor, t: tf.Tensor = tf.zeros((1, 1))) -> tf.Tensor:
        """
        Calculates the gradient of the drift wrt the states x(t).

        ..math:: df(x(t))/dx(t)

        :param x: states with shape (num_states, state_dim).
        :param t: time of states with shape (num_states, 1), defaults to zero.

        :return: the gradient of the SDE drift with shape (num_states, state_dim).
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            drift_val = self.drift(x, t)
        dfx = tape.batch_jacobian(drift_val, x)
        return dfx

    def expected_gradient_drift(self, q_mean: tf.Tensor, q_covar: tf.Tensor) -> tf.Tensor:
        """
         Calculates the Expectation of the gradient of the drift under the provided Gaussian over states

        ..math:: E_q(.)[f'(x(t))]

        :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
        :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

        :return: the expectation value with shape (n_batch, num_states, state_dim).
        """
        n_batch, n_states, state_dim = q_mean.shape
        q_mean = tf.reshape(q_mean, (-1, state_dim))
        q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))
        val = mvnquad(self.gradient_drift, q_mean, q_covar, H=10, Din=state_dim, Dout=(state_dim, state_dim))

        val = tf.reshape(val, (n_batch, n_states, state_dim, state_dim))
        return val
