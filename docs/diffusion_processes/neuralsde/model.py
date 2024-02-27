import math

import torch
import torchsde
from torch import nn, distributions
from docs.diffusion_processes.neuralsde.neural_sde_exp_utils import stable_division


class LatentSDE(torchsde.SDEIto):

    def __init__(self, x0_mu, x0_var, Q, time_grid, h):
        """

        Args:
            x0_mu: x0 mean
            x0_var: x0 variance
            Q: SDE spectral density
            h: prior drift with signature: def h(t, y)
            time_grid: time_grid [0, T]
        """
        super(LatentSDE, self).__init__(noise_type="diagonal")
        logvar = math.log(x0_var)

        self.h = h
        self.time_grid = time_grid
        self.dt = float(time_grid[1] - time_grid[0])

        self.register_buffer("mu", torch.tensor([[x0_mu]]))
        self.register_buffer("Q", torch.tensor([[Q]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[x0_mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift
        # There is one extra for x0.
        self.A = nn.Parameter(torch.randn((self.time_grid.shape[0], 1, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.randn((self.time_grid.shape[0], 1)), requires_grad=True)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[x0_mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def _find_idx_of_t(self, t):
        """
        Find idx of t in time_grid.
        """
        t = torch.round(t, decimals=3)
        return (self.time_grid == t).nonzero(as_tuple=True)

    def f(self, t, y):  # Approximate posterior drift.
        idx = self._find_idx_of_t(t)[0]
        return torch.reshape(y * self.A[idx] + self.b[idx], y.shape)

    def g(self, t, y):  # Shared diffusion.
        return self.Q.repeat(y.size(0), 1)

    # def h(t, y):  # Prior drift.
    #     return 4 * y * (1 - y**2)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)

        u = stable_division(f-h, g)
        f_logqp = .5 * (u**2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = torchsde.sdeint(
            sde=self,
            y0=aug_y0,
            ts=self.time_grid,
            method="euler",
            dt=self.dt,
            adaptive=False,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return torchsde.sdeint(self, y0, self.time_grid, method='euler', adaptive=False, dt=self.dt,
                               names={'drift': 'h'})

    def sample_q(self, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return torchsde.sdeint(self, y0, self.time_grid, method='euler', adaptive=False, dt=self.dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)

