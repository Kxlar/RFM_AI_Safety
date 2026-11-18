import torch
import torch.nn as nn
from torch.func import jvp
import math
from typing import Tuple


class SinusoidalEmbedding(nn.Module):
    """Fourier / sinusoidal scalar embedding for time t or noise scale epsilon."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeVelocityMLP(nn.Module):
    """Time-conditioned vector field network v_theta(x, t) for RFM and MFM in R^D."""

    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], -1)
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


class IntervalVelocityMLP(nn.Module):
    """Two-time average velocity network u_theta(x_t, r, t) for Riemannian MeanFlow in R^D."""

    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.r_embed = SinusoidalEmbedding(time_dim)
        self.t_embed = SinusoidalEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + 2 * time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self, x: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        r_emb = self.r_embed(r)
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x, r_emb, t_emb], dim=-1))


class MFMGeodesicInterpolantNet(nn.Module):
    """
    Metric Flow Matching Geodesic Interpolant Network (Kapusniak et al., 2024):
        gamma_eta(x0, x1, t) = (1 - t) x0 + t x1 + t (1 - t) MLP_eta(x0, x1, t)
    Uses forward-mode automatic differentiation (`torch.func.jvp`) to compute
    (gamma_eta(t), d/dt gamma_eta(t)) in O(1) pass independent of ambient dimension D!
    """

    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == 1 and x0.shape[0] > 1:
            t = t.expand(x0.shape[0], -1)
        t_emb = self.time_embed(t)
        displacement = self.net(torch.cat([x0, x1, t_emb], dim=-1))
        return (1.0 - t) * x0 + t * x1 + t * (1.0 - t) * displacement

    def pos_and_vel(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes gamma_eta(x0, x1, t) and d/dt gamma_eta(x0, x1, t) in a single O(1)
        forward-mode JVP pass regardless of dimension D.
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == 1 and x0.shape[0] > 1:
            t = t.expand(x0.shape[0], -1)

        def curve_of_t(t_in: torch.Tensor) -> torch.Tensor:
            return self.forward(x0, x1, t_in)

        ones_t = torch.ones_like(t)
        xt, dot_xt = jvp(curve_of_t, (t,), (ones_t,))
        return xt, dot_xt

    def position_and_velocity(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_and_vel(x0, x1, t)
