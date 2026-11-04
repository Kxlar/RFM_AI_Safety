import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseRiemannianMetric(ABC, nn.Module):
    """
    Abstract base class for a Riemannian metric G(x) on R^D.
    Provides both G(x) (covariant metric tensor, small in data valleys and large off-manifold)
    and Gamma(x) = G(x)^{-1} (contravariant co-metric, large along data tangent and small off-manifold).
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
    Places Gaussian kernels on data centers; G(x) is small in data valleys and 1/lambda_reg on the empty plateau.
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
            dists = torch.cdist(centers, centers)
            bandwidth = torch.median(dists[dists > 0]).item() * 0.35
        self.bandwidth = bandwidth
        self.lambda_reg = lambda_reg

        with torch.no_grad():
            d2 = torch.cdist(centers, centers) ** 2
            w_ref = torch.exp(-d2 / (2.0 * self.bandwidth ** 2)).mean(dim=-1)
            ref_max = torch.quantile(w_ref, 0.90).clamp(min=1e-6)
        self.register_buffer("ref_max", ref_max)

    def _weight_function(self, x: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(x, self.centers) ** 2  # [B, M]
        weights = torch.exp(-dist_sq / (2.0 * self.bandwidth ** 2))
        h_x = torch.clamp(weights.mean(dim=-1) / self.ref_max, min=0.0, max=1.0)
        return h_x

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        h_x = self._weight_function(x)  # [B] in [0, 1]
        scale = 1.0 / (0.15 * (h_x ** 2) + self.lambda_reg)
        eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, D, D)
        return scale.view(B, 1, 1) * eye

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        h_x = self._weight_function(x)
        scale = 0.15 * (h_x ** 2) + self.lambda_reg
        eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, D, D)
        return scale.view(B, 1, 1) * eye


class LANDMetric(BaseRiemannianMetric):
    """
    Locally Adaptive Normal Distribution (LAND) Metric (Arvanitidis et al., 2016).
    Centers the local tangent covariance around the conditional manifold expectation
    mu(x) = sum_n w_n(x) x_n to prevent off-manifold distance-squared inflation d_perp^2,
    and gates with the smooth kernel density envelope h(x)^2 so G(x) forms a deep valley
    on the data manifold and rises smoothly to (1/lambda_reg)*I in empty space.
    """
    def __init__(
        self,
        data: torch.Tensor,
        max_Reference_points: int = 1024,
        sigma: float = 0.22,
        lambda_reg: float = 1e-2,
        c_iso: float = 0.04,
    ):
        super().__init__(dim=data.shape[-1])
        idx = torch.randperm(data.shape[0])[: min(max_Reference_points, data.shape[0])]
        ref_pts = data[idx].clone().detach()
        self.register_buffer("data", ref_pts)
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.c_iso = c_iso

        with torch.no_grad():
            d2 = torch.cdist(ref_pts, ref_pts) ** 2
            w_ref = torch.exp(-d2 / (2.0 * (self.sigma * 0.85) ** 2)).mean(dim=-1)
            ref_density = torch.quantile(w_ref, 0.85).clamp(min=1e-6)
        self.register_buffer("ref_density", ref_density)

    def cometric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        diff_to_x = self.data.unsqueeze(0) - x.unsqueeze(1)  # [B, N, D]
        dist_sq = torch.sum(diff_to_x ** 2, dim=-1)          # [B, N]
        weights = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))  # [B, N]
        w_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 1. Conditional manifold mean mu(x) = E[x_data | x] eliminates off-manifold bias!
        mu_x = w_norm @ self.data  # [B, D]
        diff_centered = self.data.unsqueeze(0) - mu_x.unsqueeze(1)  # [B, N, D]

        # 2. Local tangent covariance around mu(x), scaled to dimensionless units
        Sigma_tan = torch.einsum("bn,bni,bnj->bij", w_norm, diff_centered, diff_centered)
        Sigma_tan = Sigma_tan / (self.sigma ** 2) * 0.20

        # 3. Smooth data-support density envelope h(x) in [0, 1]
        w_supp = torch.exp(-dist_sq / (2.0 * (self.sigma * 0.85) ** 2))
        h_x = torch.clamp(w_supp.mean(dim=-1) / self.ref_density, min=0.0, max=1.0).view(B, 1, 1)

        eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, D, D)
        return (h_x ** 2) * (Sigma_tan + self.c_iso * eye) + self.lambda_reg * eye

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        Gamma = self.cometric_tensor(x)
        return torch.linalg.inv(Gamma)


class CDCMetric(BaseRiemannianMetric):
    """
    Learned Carré du Champ (CDC) Riemannian Metric (Bamberger et al., 2026).
    Trains a neural network to regress the centered conditional tangent covariance
    Cov(X | Y=y) / (2*eps) on the manifold, and modulates the learned co-metric with
    a smooth support envelope h(x)^2 so that:
      - On the data manifold: Gamma(x) is large and aligned with the manifold tangent (G(x) is small -> valley).
      - Off the manifold: Gamma(x) decays smoothly to lambda_reg * I (G(x) is large -> high walls).
    """
    def __init__(
        self,
        dim: int = 2,
        rank: int = 2,
        hidden_dim: int = 128,
        eval_eps: float = 0.03,
        lambda_reg: float = 1e-2,
        c_iso: float = 0.04,
        support_sigma: float = 0.20,
    ):
        super().__init__(dim=dim)
        self.rank = rank
        self.eval_eps = eval_eps
        self.lambda_reg = lambda_reg
        self.c_iso = c_iso
        self.support_sigma = support_sigma

        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank * dim),
        )
        self.register_buffer("landmarks", torch.zeros(256, dim))
        self.register_buffer("ref_density", torch.tensor(0.1))
        self.register_buffer("has_landmarks", torch.tensor(False))

    def _support_envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Returns smooth density envelope h(x) in [0, 1] (1 on data manifold, 0 in empty space)."""
        if not self.has_landmarks.item():
            return torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        dist_sq = torch.cdist(x, self.landmarks) ** 2
        w = torch.exp(-dist_sq / (2.0 * self.support_sigma ** 2)).mean(dim=-1)
        h_x = torch.clamp(w / self.ref_density, min=0.0, max=1.0)
        return h_x.view(-1, 1, 1)

    def predict_cdc(self, y: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if eps.ndim == 1:
            eps = eps.unsqueeze(-1)
        inp = torch.cat([y, torch.log(eps + 1e-8)], dim=-1)
        m = self.net(inp).view(-1, self.rank, self.dim)
        # Bound raw network output with tanh so off-manifold MLP extrapolation is impossible
        m = 0.45 * torch.tanh(m)
        Gamma = torch.bmm(m.transpose(1, 2), m)
        return Gamma

    def fit(
        self,
        data: torch.Tensor,
        epochs: int = 800,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Trains the CDC network on centered conditional tangent covariances
        Cov(X | Y=y) / (2*eps) so normal off-manifold drift d_perp^2 never corrupts Gamma_theta.
        """
        self.to(device)
        with torch.no_grad():
            n_lm = min(256, data.shape[0])
            idx = torch.randperm(data.shape[0], device=data.device)[:n_lm]
            lm = data[idx].clone().detach().to(device)
            self.landmarks.resize_(lm.shape).copy_(lm)
            d2_lm = torch.cdist(lm, lm) ** 2
            w_lm = torch.exp(-d2_lm / (2.0 * self.support_sigma ** 2)).mean(dim=-1)
            self.ref_density.copy_(torch.quantile(w_lm, 0.85).clamp(min=1e-6))
            self.has_landmarks.fill_(True)

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data.detach()), batch_size=batch_size, shuffle=True
        )

        ref_data = data[torch.randperm(data.shape[0])[:1024]].detach().to(device)

        self.train()
        for _ in range(epochs):
            for (x,) in loader:
                x = x.to(device)
                B = x.shape[0]
                # Sample noise scales focused around the local manifold neighborhood
                log_eps = torch.randn(B, 1, device=device) * 0.6 - 3.2
                eps = torch.clamp(torch.exp(log_eps), min=2e-3, max=0.08)

                y = x + torch.sqrt(eps) * torch.randn_like(x)

                # Compute centered empirical posterior covariance E[(X - E[X|y])(X - E[X|y])^T | y] / (2*eps)
                with torch.no_grad():
                    diff_ref = ref_data.unsqueeze(0) - y.unsqueeze(1)  # [B, N_ref, D]
                    d2_ref = torch.sum(diff_ref ** 2, dim=-1)          # [B, N_ref]
                    w_post = torch.exp(-d2_ref / (2.0 * eps))          # Posterior weights p(x_n | y)
                    w_post = w_post / (w_post.sum(dim=-1, keepdim=True) + 1e-8)
                    mu_post = w_post @ ref_data                        # Conditional mean E[X | y]
                    centered = ref_data.unsqueeze(0) - mu_post.unsqueeze(1)
                    cov_post = torch.einsum("bn,bni,bnj->bij", w_post, centered, centered)
                    target = (cov_post / (2.0 * eps.unsqueeze(-1))) * 0.25

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
        eps = torch.full((B, 1), self.eval_eps, device=x.device, dtype=x.dtype)
        Gamma_raw = self.predict_cdc(x, eps)
        h_x = self._support_envelope(x)  # [B, 1, 1] in [0, 1]
        eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, D, D)
        return (h_x ** 2) * (Gamma_raw + self.c_iso * eye) + self.lambda_reg * eye

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