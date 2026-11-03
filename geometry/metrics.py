import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseRiemannianMetric(ABC, nn.Module):
    """
    Abstract base class for a Riemannian metric G(x) on R^D.
    Provides both G(x) (covariant metric tensor, large off-manifold)
    and Gamma(x) = G(x)^{-1} (contravariant co-metric, small off-manifold).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Returns G(x) of shape [B, D, D]."""
        pass

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Returns G(x)^{-1} of shape [B, D, D]."""
        G = self.metric_tensor(x)
        return torch.linalg.inv(G)

    def inner_product(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes <u, v>_{G(x)} = u^T G(x) v of shape [B]."""
        G = self.metric_tensor(x)
        return torch.einsum("bi,bij,bj->b", u, G, v)

    def norm_squared(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes ||u||_{G(x)}^2 of shape [B]."""
        return self.inner_product(x, u, u)

    def riemannian_gradient(self, x: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
        """Computes grad_g f(x) = G(x)^{-1} nabla f(x) of shape [B, D]."""
        Gamma = self.cometric_tensor(x)
        return torch.bmm(Gamma, euclidean_grad.unsqueeze(-1)).squeeze(-1)


class RBFMetric(BaseRiemannianMetric):
    """
    Isotropic/Diagonal RBF Metric (Arvanitidis et al. / Kapusniak et al., 2024).
    Places Gaussian kernels on data centers; metric is small near data and 1/lambda far away.
    """
    def __init__(
        self,
        data: torch.Tensor,
        num_centers: int = 256,
        bandwidth: Optional[float] = None,
        lambda_reg: float = 1e-2,
    ):
        super().__init__(dim=data.shape[-1])
        idx = torch.randperm(data.shape[0])[: min(num_centers, data.shape[0])]
        centers = data[idx].clone().detach()
        self.register_buffer("centers", centers)

        if bandwidth is None:
            # Median heuristic on pairwise distances
            dists = torch.cdist(centers, centers)
            bandwidth = torch.median(dists[dists > 0]).item() * 0.5
        self.bandwidth = bandwidth
        self.lambda_reg = lambda_reg

    def _weight_function(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D], centers: [M, D]
        dist_sq = torch.cdist(x, self.centers) ** 2  # [B, M]
        weights = torch.exp(-dist_sq / (2.0 * self.bandwidth ** 2))
        # Normalized local density proxy in (0, 1]
        h_x = weights.mean(dim=-1) / (weights.mean(dim=-1).max().detach() + 1e-8)
        return h_x

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        h_x = self._weight_function(x)  # [B]
        scale = 1.0 / (h_x + self.lambda_reg)  # Large when h_x -> 0
        eye = torch.eye(D, device=x.device).unsqueeze(0).expand(B, D, D)
        return scale.view(B, 1, 1) * eye

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        h_x = self._weight_function(x)
        scale = h_x + self.lambda_reg
        eye = torch.eye(D, device=x.device).unsqueeze(0).expand(B, D, D)
        return scale.view(B, 1, 1) * eye


class LANDMetric(BaseRiemannianMetric):
    """
    Locally Adaptive Normal Distribution (LAND) Metric (Arvanitidis et al., 2016).
    G(x) = (Sigma(x) + lambda I)^{-1}, where Sigma(x) is the locally weighted covariance matrix:
    Sigma_ij(x) = sum_n w_n(x) (x_n - x)_i (x_n - x)_j, w_n(x) = exp(-||x - x_n||^2 / 2*sigma^2).
    """
    def __init__(
        self,
        data: torch.Tensor,
        max_Reference_points: int = 1024,
        sigma: float = 0.35,
        lambda_reg: float = 1e-2,
    ):
        super().__init__(dim=data.shape[-1])
        idx = torch.randperm(data.shape[0])[: min(max_Reference_points, data.shape[0])]
        self.register_buffer("data", data[idx].clone().detach())
        self.sigma = sigma
        self.lambda_reg = lambda_reg

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # Sigma(x) + lambda I acts directly as the contravariant co-metric!
        B, D = x.shape
        diff = self.data.unsqueeze(0) - x.unsqueeze(1)  # [B, N, D]
        dist_sq = torch.sum(diff ** 2, dim=-1)          # [B, N]
        weights = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))  # [B, N]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)

        # Local covariance: [B, D, D]
        Sigma = torch.einsum("bn,bni,bnj->bij", weights, diff, diff)
        eye = torch.eye(D, device=x.device).unsqueeze(0).expand(B, D, D)
        return Sigma + self.lambda_reg * eye

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        Sigma_reg = self.cometric_tensor(x)
        return torch.linalg.inv(Sigma_reg)


class CDCMetric(BaseRiemannianMetric):
    """
    Learned Carré du Champ (CDC) Riemannian Metric (Bamberger et al., 2026).
    Trains a neural network to regress the conditional CDC co-metric Gamma_eps(x),
    and provides G(x) = (Gamma_eps(x) + lambda I)^{-1} for MFM/RFM.
    """
    def __init__(
        self,
        dim: int = 2,
        rank: int = 2,
        hidden_dim: int = 128,
        eval_eps: float = 0.05,
        lambda_reg: float = 1e-2,
    ):
        super().__init__(dim=dim)
        self.rank = rank
        self.eval_eps = eval_eps
        self.lambda_reg = lambda_reg

        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank * dim),
        )

    def predict_cdc(self, y: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if eps.ndim == 1:
            eps = eps.unsqueeze(-1)
        inp = torch.cat([y, torch.log(eps + 1e-8)], dim=-1)
        m = self.net(inp).view(-1, self.rank, self.dim)
        Gamma = torch.bmm(m.transpose(1, 2), m)
        return Gamma

    def fit(
        self,
        data: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """Algorithm 2: Conditional Riemannian Metric Matching training."""
        self.to(device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
        )

        self.train()
        for _ in range(epochs):
            for (x,) in loader:
                x = x.to(device)
                B = x.shape[0]
                log_eps = torch.randn(B, 1, device=device) * 1.0 - 2.0
                eps = torch.clamp(torch.exp(log_eps), min=1e-3, max=0.5)

                y = x + torch.sqrt(eps) * torch.randn_like(x)
                delta = x - y
                target = torch.bmm(delta.unsqueeze(-1), delta.unsqueeze(1)) / (2.0 * eps.unsqueeze(-1))

                pred = self.predict_cdc(y, eps)
                loss = torch.mean(torch.sum((pred - target) ** 2, dim=(1, 2)))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
            sched.step()
        self.eval()

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        eps = torch.full((B, 1), self.eval_eps, device=x.device)
        Gamma = self.predict_cdc(x, eps)
        eye = torch.eye(D, device=x.device).unsqueeze(0).expand(B, D, D)
        return Gamma + self.lambda_reg * eye

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        Gamma_reg = self.cometric_tensor(x)
        return torch.linalg.inv(Gamma_reg)


class ConformalObstacleMetric(BaseRiemannianMetric):
    """
    Wraps any base Riemannian metric G_data(x) with a conformal obstacle barrier:
    G_tilde(x) = (1 + beta * V(x)) G_data(x).
    """
    def __init__(self, base_metric: BaseRiemannianMetric, potential_fn, beta: float = 10.0):
        super().__init__(dim=base_metric.dim)
        self.base_metric = base_metric
        self.potential_fn = potential_fn
        self.beta = beta

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        G_base = self.base_metric.metric_tensor(x)
        V = self.potential_fn.pointwise(x)  # [B]
        factor = (1.0 + self.beta * V).view(-1, 1, 1)
        return factor * G_base

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        Gamma_base = self.base_metric.cometric_tensor(x)
        V = self.potential_fn.pointwise(x)  # [B]
        factor = (1.0 / (1.0 + self.beta * V)).view(-1, 1, 1)
        return factor * Gamma_base