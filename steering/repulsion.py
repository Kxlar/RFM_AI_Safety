import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

from geometry.metrics import BaseRiemannianMetric
from geometry.premetrics import BasePremetric


class CompactForbiddenPotential(nn.Module):
    """
    Smooth C^2 Wendland bump potential V(x) centered at `center`.
    Evaluates distances using any BasePremetric (e.g., MFMGeodesicPremetric,
    SpectralLaplacianPremetric, or EuclideanPremetric).
    Strictly satisfies V(x) = 0 and grad V(x) = 0 for all x with d(x, center) >= r_max.
    """
    def __init__(
        self,
        center: torch.Tensor,
        r_min: float = 0.25,
        r_max: float = 0.70,
        scale: float = 20.0,
        premetric: Optional[BasePremetric] = None,
    ):
        super().__init__()
        self.register_buffer("center", center.clone().detach().view(1, -1))
        self.r_min = r_min
        self.r_max = r_max
        self.scale = scale
        self.premetric = premetric

    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        target = self.center.expand(x.shape[0], -1)
        if self.premetric is not None:
            return self.premetric.distance(x, target)
        return torch.linalg.norm(x - target, dim=-1)

    def pointwise(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the scalar potential V(x) for each point in batch [B]."""
        d = self.compute_distance(x)
        u = torch.clamp((d - self.r_min) / (self.r_max - self.r_min + 1e-6), min=0.0, max=1.0)
        bump = ((1.0 - u) ** 4) * (4.0 * u + 1.0)
        return self.scale * bump

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pointwise(x)


class RiemannianRepulsionStreamer:
    """
    Inference-time steering wrapper with Geodesic-scaled Repulsion, Control Barrier Filtering (CBF),
    and Gyroscopic Circulation (to eliminate stuck saddle points).
    """
    def __init__(
        self,
        flow_model: nn.Module,
        potential: Optional[CompactForbiddenPotential] = None,
        metric: Optional[BaseRiemannianMetric] = None,
        alpha: float = 1.0,
        gyro_curl: float = 0.4,
        use_cbf_filter: bool = True,
    ):
        self.flow_model = flow_model
        self.potential = potential
        self.metric = metric
        self.alpha = alpha
        self.gyro_curl = gyro_curl
        self.use_cbf_filter = use_cbf_filter

    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        B, D = x.shape
        t_tensor = torch.full((B, 1), t, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            v_base = self.flow_model.velocity(x, t_tensor)

        if self.potential is None or self.alpha == 0.0:
            return v_base

        # Compute Euclidean gradient of the compact geodesic potential V(x)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            V_val = self.potential.pointwise(x_in).sum()
            grad_euc = torch.autograd.grad(V_val, x_in)[0]  # [B, D]

        # Map to intrinsic Riemannian gradient: grad_g V(x) = G(x)^{-1} grad_euc V(x)
        if self.metric is not None:
            grad_g = self.metric.riemannian_gradient(x, grad_euc)
            if self.use_cbf_filter:
                inward_proj = self.metric.inner_product(x, v_base, grad_g).unsqueeze(-1)
                grad_norm_sq = self.metric.norm_squared(x, grad_g).unsqueeze(-1) + 1e-6
                cbf_gain = 1.0 + torch.relu(inward_proj) / grad_norm_sq
                repulsion = self.alpha * cbf_gain * grad_g
            else:
                repulsion = self.alpha * grad_g
        else:
            grad_g = grad_euc
            if self.use_cbf_filter:
                inward_proj = torch.sum(v_base * grad_g, dim=-1, keepdim=True)
                grad_norm_sq = torch.sum(grad_g ** 2, dim=-1, keepdim=True) + 1e-6
                cbf_gain = 1.0 + torch.relu(inward_proj) / grad_norm_sq
                repulsion = self.alpha * cbf_gain * grad_g
            else:
                repulsion = self.alpha * grad_g

        # Gyroscopic circulation term orthogonal to grad_g V(x) to prevent stalling
        circulation = torch.zeros_like(grad_g)
        if D == 2 and self.gyro_curl != 0.0:
            circulation = torch.stack([-grad_g[:, 1], grad_g[:, 0]], dim=-1) * self.gyro_curl

        return v_base - repulsion + circulation


@torch.no_grad()
def integrate_ode(
    field_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 100,
    solver: str = "rk4",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrates dx/dt = field_fn(x, t) over t in [0, 1]. Returns (x1, trajectory [T+1, B, D])."""
    dt = 1.0 / steps
    x = x0.clone()
    traj = [x.clone()]

    for i in range(steps):
        t = i * dt
        if solver == "euler":
            x = x + dt * field_fn(x, t)
        elif solver == "rk4":
            k1 = field_fn(x, t)
            k2 = field_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = field_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = field_fn(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown ODE solver: {solver}")
        traj.append(x.clone())

    return x, torch.stack(traj, dim=0)