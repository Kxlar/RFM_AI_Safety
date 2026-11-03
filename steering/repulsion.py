import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

from geometry.metrics import BaseRiemannianMetric
from geometry.premetrics import BasePremetric


class CompactForbiddenPotential(nn.Module):
    """
    Smooth C^2 Wendland bump potential V(x) centered at `center`.
    Automatically calibrates non-Euclidean premetrics (Diffusion, Biharmonic, Geodesic)
    so that `r_min` and `r_max` correspond to consistent physical radii around `center`.
    Strictly satisfies V(x) = 0 and grad V(x) = 0 for all x with calibrated d_g(x, center) >= r_max.
    """
    def __init__(
        self,
        center: torch.Tensor,
        r_min: float = 0.30,
        r_max: float = 0.65,
        scale: float = 2.5,
        premetric: Optional[BasePremetric] = None,
    ):
        super().__init__()
        self.register_buffer("center", center.clone().detach().view(1, -1))
        self.r_min = r_min
        self.r_max = r_max
        self.scale = scale
        self._premetric = None
        self.register_buffer("dist_scale", torch.tensor(1.0, device=center.device))
        self.premetric = premetric

    @property
    def premetric(self) -> Optional[BasePremetric]:
        return self._premetric

    @premetric.setter
    def premetric(self, pm: Optional[BasePremetric]):
        self._premetric = pm
        if pm is not None:
            self._calibrate_distance_scale()
        else:
            self.dist_scale.fill_(1.0)

    @torch.no_grad()
    def _calibrate_distance_scale(self):
        """
        Samples a ring of points at Euclidean distance `r_min` around `center`
        and computes the scaling factor so median(d_premetric(ring, center)) == r_min.
        Prevents spectral/biharmonic distances from bleeding across the entire manifold.
        """
        device = self.center.device
        angles = torch.linspace(0, 2 * torch.pi, 64, device=device)[:-1]
        ring = self.center + self.r_min * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        raw_dists = self._premetric.distance(ring, self.center.expand_as(ring))
        median_d = torch.median(raw_dists).clamp(min=1e-6)
        self.dist_scale.copy_(self.r_min / median_d)

    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        target = self.center.expand(x.shape[0], -1)
        if self._premetric is not None:
            return self._premetric.distance(x, target) * self.dist_scale
        return torch.linalg.norm(x - target, dim=-1)

    def pointwise(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the scalar potential V(x) in [0, scale] for each point in batch [B]."""
        d = self.compute_distance(x)
        u = torch.clamp((d - self.r_min) / (self.r_max - self.r_min + 1e-6), min=0.0, max=1.0)
        # Wendland C^2 compact polynomial: (1 - u)^4 * (4u + 1) -> strictly 0 for d >= r_max
        bump = ((1.0 - u) ** 4) * (4.0 * u + 1.0)
        return self.scale * bump

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pointwise(x)


class RiemannianRepulsionStreamer:
    """
    Geodesic-Respecting Control Barrier & Gyroscopic Steering Wrapper:
      1. Zero-Interference on Safe Geodesics: If the base flow v_base(x, t) is moving away from
         or tangential to the forbidden zone (<v_base, -grad_g V>_g <= 0) outside the core,
         the repulsive force cancels to strictly ZERO.
      2. Velocity-Proportional Deflection: When v_base points into the forbidden zone, the inward
         normal component <v_base, n_g>_g is cancelled and smoothly deflected along the manifold
         tangent via the g-orthogonal circulation operator J_g, avoiding explosive kicks.
    """
    def __init__(
        self,
        flow_model: nn.Module,
        potential: Optional[CompactForbiddenPotential] = None,
        metric: Optional[BaseRiemannianMetric] = None,
        alpha: float = 1.2,
        gyro_curl: float = 0.6,
    ):
        self.flow_model = flow_model
        self.potential = potential
        self.metric = metric
        self.alpha = alpha
        self.gyro_curl = gyro_curl

    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        B, D = x.shape
        t_tensor = torch.full((B, 1), t, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            v_base = self.flow_model.velocity(x, t_tensor)

        if self.potential is None or self.alpha == 0.0:
            return v_base

        # Compute Euclidean gradient and value of the compact geodesic potential V(x)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            V_pts = self.potential.pointwise(x_in)  # [B] in [0, scale]
            grad_euc = torch.autograd.grad(V_pts.sum(), x_in)[0].detach()  # [B, D]
            V_pts = V_pts.detach()

        # Normalized barrier intensity b(x) in [0, 1] (0 outside r_max, 1 inside r_min)
        barrier_intensity = (V_pts / (self.potential.scale + 1e-8)).unsqueeze(-1)  # [B, 1]

        # Active mask: only compute steering where V(x) > 0 (d_g(x, center) < r_max)
        if self.metric is not None:
            grad_g = self.metric.riemannian_gradient(x, grad_euc)
            grad_norm_g = torch.sqrt(self.metric.norm_squared(x, grad_g) + 1e-8).unsqueeze(-1)
            # Unit inward normal n_in pointing toward the forbidden center (uphill in V)
            n_in = grad_g / grad_norm_g
            # Component of v_base pointing INTO the forbidden zone: <v_base, n_in>_G
            inward_speed = self.metric.inner_product(x, v_base, n_in).unsqueeze(-1)
            v_speed = torch.sqrt(self.metric.norm_squared(x, v_base) + 1e-8).unsqueeze(-1)
        else:
            grad_g = grad_euc
            grad_norm_euc = torch.linalg.norm(grad_g, dim=-1, keepdim=True) + 1e-8
            n_in = grad_g / grad_norm_euc
            inward_speed = torch.sum(v_base * n_in, dim=-1, keepdim=True)
            v_speed = torch.linalg.norm(v_base, dim=-1, keepdim=True) + 1e-8

        # 1. Cancel ONLY positive inward velocity (<v_base, n_in>_g > 0) + soft emergency push if already inside core
        inward_component = torch.relu(inward_speed)
        core_escape = 0.5 * v_speed * (barrier_intensity ** 2)
        normal_repulsion = self.alpha * barrier_intensity * (inward_component + core_escape) * n_in

        # 2. Tangential circulation (slides around the boundary only when heading into the obstacle!)
        circulation = torch.zeros_like(v_base)
        if D == 2 and self.gyro_curl != 0.0:
            # Choose circulation sign dynamically to match the particle's existing tangential preference
            t_vec = torch.stack([-n_in[:, 1], n_in[:, 0]], dim=-1)
            tangential_dot = torch.sum(v_base * t_vec, dim=-1, keepdim=True)
            sign = torch.where(tangential_dot >= 0, 1.0, -1.0)
            circulation = sign * self.gyro_curl * barrier_intensity * inward_component * t_vec

        return v_base - normal_repulsion + circulation


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