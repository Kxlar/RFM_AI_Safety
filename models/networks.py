import torch
import torch.nn as nn
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
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeVelocityMLP(nn.Module):
    """Time-conditioned vector field network v_theta(x, t) for RFM and MFM."""
    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
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
    """Two-time average velocity network u_theta(x_t, r, t) for Riemannian MeanFlow."""
    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
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

    def forward(self, x: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
    """
    def __init__(self, dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
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

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        """Returns both gamma_eta(x0, x1, t) and its exact time derivative d/dt gamma_eta."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        with torch.enable_grad():
            t_req = t.detach().requires_grad_(True)
            xt = self.forward(x0, x1, t_req)
            grads = [
                torch.autograd.grad(xt[:, d].sum(), t_req, create_graph=True)[0]
                for d in range(xt.shape[-1])
            ]
            dot_xt = torch.cat(grads, dim=-1)
        return xt, dot_xt

    def position_and_velocity(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_and_vel(x0, x1, t)