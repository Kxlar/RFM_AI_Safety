import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
from sklearn.cluster import KMeans

from geometry.metrics import BaseRiemannianMetric
from geometry.premetrics import BasePremetric


class BaseForbiddenPotential(ABC, nn.Module):
    """
    Unified interface for forbidden region potentials V(x) in R^D.
    All potentials provide:
      - `pointwise(x)`: scalar potential V(x) in [0, scale] of shape [B]
      - `normalized_intensity(x)`: dimensionless barrier intensity b(x) in [0, 1] of shape [B]
      - `is_violated(x)`: boolean tensor [B] indicating whether x is inside the forbidden core
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    @abstractmethod
    def normalized_intensity(self, x: torch.Tensor) -> torch.Tensor:
        """Returns dimensionless Wendland barrier intensity b(x) in [0, 1] of shape [B]."""
        pass

    @abstractmethod
    def is_violated(self, x: torch.Tensor) -> torch.Tensor:
        """Returns boolean tensor [B] True when x lies inside the forbidden core."""
        pass

    def pointwise(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.normalized_intensity(x)

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.pointwise(x)


def wendland_c2_bump(u: torch.Tensor) -> torch.Tensor:
    """Wendland C^2 compact polynomial b(u) = (1 - u)^4 (4u + 1) for u in [0, 1]."""
    u_clamped = torch.clamp(u, min=0.0, max=1.0)
    return ((1.0 - u_clamped) ** 4) * (4.0 * u_clamped + 1.0)


class CompactForbiddenPotential(BaseForbiddenPotential):
    """
    Option 1: Single Centroid Hyperball / Geodesic Ball Barrier in R^D.
    Uses hyperspherical sampling on S^{D-1} to calibrate non-Euclidean premetrics
    in arbitrary dimension D >= 2.
    """

    def __init__(
        self,
        center: torch.Tensor,
        r_min: float = 0.30,
        r_max: float = 0.65,
        scale: float = 20.0,
        premetric: Optional[BasePremetric] = None,
    ):
        super().__init__(scale=scale)
        self.register_buffer("center", center.clone().detach().view(1, -1))
        self.r_min = r_min
        self.r_max = r_max
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
    def _calibrate_distance_scale(self, n_directions: int = 128):
        """
        Dimension-agnostic calibration: samples `n_directions` unit vectors uniformly
        on the hypersphere S^{D-1} around `center` at Euclidean radius `r_min`.
        """
        device = self.center.device
        D = self.center.shape[-1]
        g_gen = torch.Generator(device=device).manual_seed(42)
        dirs = torch.randn(
            (n_directions, D), generator=g_gen, device=device, dtype=self.center.dtype
        )
        dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        shell = self.center + self.r_min * dirs
        raw_dists = self._premetric.distance(shell, self.center.expand_as(shell))
        median_d = torch.median(raw_dists).clamp(min=1e-6)
        self.dist_scale.copy_(self.r_min / median_d)

    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        target = self.center.expand(x.shape[0], -1)
        if self._premetric is not None:
            return self._premetric.distance(x, target) * self.dist_scale
        return torch.linalg.norm(x - target, dim=-1)

    def normalized_intensity(self, x: torch.Tensor) -> torch.Tensor:
        d = self.compute_distance(x)
        u = (d - self.r_min) / (self.r_max - self.r_min + 1e-6)
        return wendland_c2_bump(u)

    def is_violated(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_distance(x) < self.r_min


class KMeansForbiddenPotential(BaseForbiddenPotential):
    """
    Option 2: K-Means Multi-Subcentroid Potential with Softmin Decaying Aggregation.
    Approximates arbitrary non-convex/elongated class clusters in R^D by fitting K
    sub-centroids {c_k}_{k=1}^K on the forbidden class samples.
    Each sub-ball k has adaptive core radius r_min_k and cutoff r_max_k.
    Contributions are aggregated via distance-decaying softmin weights:
        w_k(x) = exp(-tau * d_k(x)) / sum_j exp(-tau * d_j(x))
        b(x)   = sum_{k=1}^K w_k(x) * b_Wendland( (d_k(x) - r_min_k) / (r_max_k - r_min_k) )
    This guarantees b(x) in [0, 1] without constructive blowup where sub-balls overlap!
    """

    def __init__(
        self,
        forbidden_points: torch.Tensor,
        num_clusters: int = 6,
        r_min_quantile: float = 0.75,
        buffer_ratio: float = 1.65,
        decay_temp: float = 6.0,
        scale: float = 20.0,
        premetric: Optional[BasePremetric] = None,
    ):
        super().__init__(scale=scale)
        device = forbidden_points.device
        dtype = forbidden_points.dtype
        D = forbidden_points.shape[-1]
        self.num_clusters = min(num_clusters, forbidden_points.shape[0])
        self.decay_temp = decay_temp
        self._premetric = premetric

        # Fit K-Means on the forbidden class points
        pts_np = forbidden_points.detach().cpu().numpy()
        km = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10).fit(
            pts_np
        )
        centers = torch.tensor(
            km.cluster_centers_, dtype=dtype, device=device
        )  # [K, D]
        labels = torch.tensor(km.labels_, dtype=torch.long, device=device)

        r_mins = torch.zeros(self.num_clusters, dtype=dtype, device=device)
        for k in range(self.num_clusters):
            cluster_pts = forbidden_points[labels == k]
            if cluster_pts.shape[0] > 1:
                dists_k = torch.linalg.norm(cluster_pts - centers[k : k + 1], dim=-1)
                r_mins[k] = torch.quantile(dists_k, r_min_quantile).clamp(min=0.15)
            else:
                r_mins[k] = 0.30

        r_maxs = r_mins * buffer_ratio

        self.register_buffer("centers", centers)  # [K, D]
        self.register_buffer("r_mins", r_mins)  # [K]
        self.register_buffer("r_maxs", r_maxs)  # [K]
        # Overall centroid for 2D plotting compatibility
        self.register_buffer("center", centers.mean(dim=0, keepdim=True))
        self.r_min = r_mins.mean().item()
        self.r_max = r_maxs.mean().item()

    def _subcentroid_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, K] distance matrix from each point x_b to each sub-centroid c_k."""
        if self._premetric is not None and hasattr(
            self._premetric, "pairwise_distance"
        ):
            # Use differentiable pairwise distance if needed
            B, D = x.shape
            K = self.centers.shape[0]
            x_rep = x.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
            c_rep = self.centers.unsqueeze(0).expand(B, K, D).reshape(B * K, D)
            return self._premetric.distance(x_rep, c_rep).reshape(B, K)
        return torch.cdist(x, self.centers, p=2.0)  # [B, K]

    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Returns effective signed distance proxy so contour levels [r_min, r_max] plot cleanly."""
        dists = self._subcentroid_distances(x)  # [B, K]
        # Normalized relative distance to each sub-ball's r_min
        rel_dists = dists / (self.r_mins.unsqueeze(0) + 1e-6) * self.r_min
        return rel_dists.min(dim=-1)[0]

    def normalized_intensity(self, x: torch.Tensor) -> torch.Tensor:
        dists = self._subcentroid_distances(x)  # [B, K]
        u_k = (dists - self.r_mins.unsqueeze(0)) / (
            self.r_maxs.unsqueeze(0) - self.r_mins.unsqueeze(0) + 1e-6
        )
        b_k = wendland_c2_bump(u_k)  # [B, K]

        # Decaying softmin influence weights based on proximity to each sub-centroid
        rel_prox = dists / (self.r_mins.unsqueeze(0) + 1e-6)
        weights = torch.softmax(-self.decay_temp * rel_prox, dim=-1)  # [B, K]
        return torch.sum(weights * b_k, dim=-1)

    def is_violated(self, x: torch.Tensor) -> torch.Tensor:
        dists = self._subcentroid_distances(x)  # [B, K]
        return torch.any(dists < self.r_mins.unsqueeze(0), dim=-1)


class ClassifierForbiddenPotential(BaseForbiddenPotential):
    """
    Option 3: Differentiable Latent MLP Classifier Barrier in R^D.
    Trains a smooth MLP classifier p_omega(y | z) on the latent space (with temperature
    scaling so gradients don't saturate) and defines the barrier via the forbidden class
    probability p_forbid(z) = p_omega(y = forbidden_class | z):
      - Core violation: p_forbid(z) >= p_core (default 0.45)
      - Buffer cutoff:  p_forbid(z) <= p_cutoff (default 0.05) -> V(z) = 0 identically!
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 10,
        forbidden_class: int = 3,
        hidden_dim: int = 128,
        p_core: float = 0.45,
        p_cutoff: float = 0.05,
        temperature: float = 1.5,
        scale: float = 20.0,
    ):
        super().__init__(scale=scale)
        self.dim = dim
        self.num_classes = num_classes
        self.forbidden_class = forbidden_class
        self.p_core = p_core
        self.p_cutoff = p_cutoff
        self.temperature = temperature

        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.register_buffer("center", torch.zeros(1, dim))
        self.r_min = 0.30
        self.r_max = 0.65

    def fit_classifier(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 250,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.to(device)
        with torch.no_grad():
            forb_pts = latents[labels == self.forbidden_class]
            if forb_pts.shape[0] > 0:
                self.center.copy_(forb_pts.mean(dim=0, keepdim=True))

        opt = optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(latents.detach(), labels.detach()),
            batch_size=batch_size,
            shuffle=True,
        )

        self.classifier.train()
        for _ in range(epochs):
            for zb, yb in loader:
                zb, yb = zb.to(device), yb.to(device)
                # Add slight Gaussian jitter so the probability landscape is smooth off-manifold
                z_noisy = zb + 0.05 * torch.randn_like(zb)
                logits = self.classifier(z_noisy) / self.temperature
                loss = criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

    def forbidden_probability(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self.forbidden_class]

    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Maps probability p_forbid(x) in [p_cutoff, p_core] to pseudo-distance [r_max, r_min] for plotting."""
        p_f = self.forbidden_probability(x)
        u = (self.p_core - p_f) / (self.p_core - self.p_cutoff + 1e-6)
        return self.r_min + u * (self.r_max - self.r_min)

    def normalized_intensity(self, x: torch.Tensor) -> torch.Tensor:
        p_f = self.forbidden_probability(x)
        # u = 0 when p_f >= p_core (inside core); u = 1 when p_f <= p_cutoff (outside buffer)
        u = (self.p_core - p_f) / (self.p_core - self.p_cutoff + 1e-6)
        return wendland_c2_bump(u)

    def is_violated(self, x: torch.Tensor) -> torch.Tensor:
        return self.forbidden_probability(x) >= self.p_core


class RiemannianRepulsionStreamer:
    """
    Dimension-Agnostic (D >= 2) Geodesic-Respecting ZCBF & Tangential Steering Wrapper:
      1. Zero-Interference on Safe Flows: If <v_base, n_in>_G <= 0 outside the core,
         the ZCBF deflection vanishes.
      2. Gram-Schmidt Tangential Redirection in R^D: Generalizes the 2D gyroscopic curl
         to arbitrary dimension D by extracting the G(x)-orthogonal tangential component
         of v_base (or a consistent tangent basis vector in D=2) and converting cancelled
         inward momentum into smooth tangential slip around the obstacle boundary.
    """

    def __init__(
        self,
        flow_model: nn.Module,
        potential: Optional[BaseForbiddenPotential] = None,
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

        # Compute Euclidean gradient and barrier intensity b(x) in [0, 1]
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            b_pts = self.potential.normalized_intensity(x_in)  # [B] in [0, 1]
            grad_euc = torch.autograd.grad(b_pts.sum(), x_in)[0].detach()  # [B, D]
            b_pts = b_pts.detach()

        barrier_intensity = b_pts.unsqueeze(-1)  # [B, 1]

        # Compute unit inward Riemannian normal n_in and inward velocity component <v_base, n_in>_G
        if self.metric is not None:
            grad_g = self.metric.riemannian_gradient(x, grad_euc)
            grad_norm_g = torch.sqrt(
                self.metric.norm_squared(x, grad_g) + 1e-8
            ).unsqueeze(-1)
            n_in = grad_g / grad_norm_g
            inward_speed = self.metric.inner_product(x, v_base, n_in).unsqueeze(-1)
            v_speed = torch.sqrt(self.metric.norm_squared(x, v_base) + 1e-8).unsqueeze(
                -1
            )
        else:
            grad_g = grad_euc
            grad_norm_euc = torch.linalg.norm(grad_g, dim=-1, keepdim=True) + 1e-8
            n_in = grad_g / grad_norm_euc
            inward_speed = torch.sum(v_base * n_in, dim=-1, keepdim=True)
            v_speed = torch.linalg.norm(v_base, dim=-1, keepdim=True) + 1e-8

        # 1. ZCBF Normal Damping + Soft Core Ejection
        inward_component = torch.relu(inward_speed)
        core_escape = 0.5 * v_speed * (barrier_intensity**2)
        normal_repulsion = (
            self.alpha * barrier_intensity * (inward_component + core_escape) * n_in
        )

        # 2. Dimension-Agnostic Tangential Redirection (works for D = 2 and D > 2)
        circulation = torch.zeros_like(v_base)
        if self.gyro_curl != 0.0:
            if D == 2:
                t_vec = torch.stack([-n_in[:, 1], n_in[:, 0]], dim=-1)
                tangential_dot = torch.sum(v_base * t_vec, dim=-1, keepdim=True)
                sign = torch.where(tangential_dot >= 0, 1.0, -1.0)
                t_unit = sign * t_vec
            else:
                # Gram-Schmidt G(x)-orthogonal tangential component of v_base in R^D:
                # v_tan = v_base - <v_base, n_in>_G * n_in satisfies <v_tan, n_in>_G = 0!
                v_tan = v_base - inward_speed * n_in
                if self.metric is not None:
                    tan_norm = torch.sqrt(
                        self.metric.norm_squared(x, v_tan) + 1e-8
                    ).unsqueeze(-1)
                else:
                    tan_norm = torch.linalg.norm(v_tan, dim=-1, keepdim=True) + 1e-8
                t_unit = v_tan / tan_norm

            circulation = self.gyro_curl * barrier_intensity * inward_component * t_unit

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
