import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePremetric(ABC, nn.Module):
    """
    Premetric d: M x M -> R_+ satisfying:
    (i) d(x, y) >= 0, (ii) d(x, y) = 0 iff x = y, (iii) grad_x d(x, y) != 0 for x != y.
    """
    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes d(x, y) for paired batches x, y of shape [B, D] -> [B]."""
        pass

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes pairwise matrix d(x_i, y_j) of shape [N, M]."""
        N, M = x.shape[0], y.shape[0]
        x_exp = x.unsqueeze(1).expand(N, M, -1).reshape(N * M, -1)
        y_exp = y.unsqueeze(0).expand(N, M, -1).reshape(N * M, -1)
        return self.distance(x_exp, y_exp).view(N, M)


class EuclideanPremetric(BasePremetric):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + self.eps)

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=2.0)


class SpectralLaplacianPremetric(BasePremetric):
    """
    Spectral distance based on Laplace-Beltrami eigenfunctions (RFM Paper / Notes Page 2):
    d_w(x, y)^2 = sum_{i=1}^K w(lambda_i) (phi_i(x) - phi_i(y))^2
    - mode == 'diffusion':  w(lambda) = exp(-2 * tau * lambda)
    - mode == 'biharmonic': w(lambda) = lambda^{-2}
    Uses Differentiable Nyström Extension so phi_i(x) is differentiable anywhere in R^D.
    """
    def __init__(
        self,
        data: torch.Tensor,
        mode: str = "diffusion",
        num_landmarks: int = 512,
        num_eigen: int = 32,
        sigma: float = 0.35,
        tau: float = 1.0,
        ambient_weight: float = 0.05,
    ):
        super().__init__()
        assert mode in ["diffusion", "biharmonic"]
        self.mode = mode
        self.sigma = sigma
        self.tau = tau
        self.ambient_weight = ambient_weight

        with torch.no_grad():
            # Subsample landmarks from data
            idx = torch.randperm(data.shape[0])[: min(num_landmarks, data.shape[0])]
            landmarks = data[idx].clone().detach()

            # Build symmetric normalized Graph Laplacian L_sym = I - D^{-1/2} K D^{-1/2}
            dist_sq = torch.cdist(landmarks, landmarks) ** 2
            K_mat = torch.exp(-dist_sq / (2.0 * sigma ** 2))
            deg = K_mat.sum(dim=-1) + 1e-8
            deg_inv_sqrt = torch.pow(deg, -0.5)
            S = deg_inv_sqrt.unsqueeze(1) * K_mat * deg_inv_sqrt.unsqueeze(0)

            # Eigendecomposition of S
            evals_S, evecs_S = torch.linalg.eigh(S)
            evals_S = torch.flip(evals_S, dims=[0])
            evecs_S = torch.flip(evecs_S, dims=[1])

            # Skip trivial constant eigenvector i=0
            k_use = min(num_eigen, landmarks.shape[0] - 1)
            mu_i = torch.clamp(evals_S[1 : k_use + 1], min=1e-4, max=0.9999)
            lambda_i = (1.0 - mu_i) / (sigma ** 2)
            u_i = evecs_S[:, 1 : k_use + 1]  # [M, K]

            if mode == "diffusion":
                w_i = torch.exp(-2.0 * tau * lambda_i)
            else:  # biharmonic
                w_i = 1.0 / (lambda_i ** 2 + 1e-4)
                w_i = w_i / w_i.max()

            self.register_buffer("landmarks", landmarks.detach())
            self.register_buffer("deg_landmarks", deg.detach())
            self.register_buffer("mu_i", mu_i.detach())
            self.register_buffer("u_i", u_i.detach())
            self.register_buffer("w_i", w_i.detach())

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Nyström extension phi(x) for arbitrary x in R^D."""
        dist_sq = torch.cdist(x, self.landmarks) ** 2  # [B, M]
        k_x = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        d_x = k_x.sum(dim=-1, keepdim=True) + 1e-8
        s_x = k_x / torch.sqrt(d_x * self.deg_landmarks.unsqueeze(0))  # [B, M]
        phi_x = (s_x @ self.u_i) / self.mu_i.unsqueeze(0)              # [B, K]
        return phi_x

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi_x = self.embed(x)
        phi_y = self.embed(y)
        spec_dist_sq = torch.sum(self.w_i.unsqueeze(0) * (phi_x - phi_y) ** 2, dim=-1)
        # Add small ambient Euclidean regularization to ensure non-degeneracy (iii) off-manifold
        euc_dist_sq = torch.sum((x - y) ** 2, dim=-1)
        return torch.sqrt(spec_dist_sq + self.ambient_weight * euc_dist_sq + 1e-6)

    @torch.no_grad()
    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fast O(N + M) spectral embedding + cdist for Optimal Transport matrices."""
        w_sqrt = torch.sqrt(self.w_i).unsqueeze(0)
        phi_x = self.embed(x) * w_sqrt
        phi_y = self.embed(y) * w_sqrt
        spec_dist_sq = torch.cdist(phi_x, phi_y, p=2.0) ** 2
        euc_dist_sq = torch.cdist(x, y, p=2.0) ** 2
        return torch.sqrt(spec_dist_sq + self.ambient_weight * euc_dist_sq + 1e-6)


class MFMGeodesicPremetric(BasePremetric):
    """
    Approximates the true Riemannian Geodesic Distance d_g(x, y) using a trained
    MFM geodesic interpolant network gamma_eta(x, y, t).
    - `distance(x, y)`: Uses exact autograd tangent velocity `pos_and_vel` + Composite
      Trapezoidal Rule O(N^{-2}) for accurate spatial gradients in steering and RFM.
    - `pairwise_distance(x, y)`: Uses chunked/vectorized Midpoint Finite-Differences
      under `@torch.no_grad()` for 10x faster Optimal Transport cost matrices.
    """
    def __init__(self, mfm_interpolant, metric: nn.Module, num_quad_steps: int = 16, ot_chunk_size: int = 8192):
        super().__init__()
        self.interpolant = mfm_interpolant
        self.metric = metric
        self.num_quad_steps = max(num_quad_steps, 4)
        self.ot_chunk_size = ot_chunk_size

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Exact autograd tangent velocity + Composite Trapezoidal quadrature (O(dt^2)).
        Differentiable wrt x and y for steering potentials and RFM targets.
        """
        B = x.shape[0]
        N = self.num_quad_steps
        t_grid = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype)
        dt = 1.0 / (N - 1)

        total_len = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i, t_val in enumerate(t_grid):
            # Trapezoidal endpoint weights: 0.5 at t=0 and t=1, 1.0 in the interior
            weight = 0.5 * dt if (i == 0 or i == N - 1) else dt
            t_batch = torch.full((B, 1), t_val.item(), device=x.device, dtype=x.dtype)
            xt, dxt = self.interpolant.pos_and_vel(x, y, t_batch)
            speed = torch.sqrt(self.metric.norm_squared(xt, dxt) + 1e-6)
            total_len = total_len + weight * speed

        return total_len

    @torch.no_grad()
    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fast graph-free Composite Midpoint Finite-Difference quadrature (O(dt^2))
        for computing [N_x, N_y] Optimal Transport cost matrices without autograd overhead.
        """
        Nx, Ny = x.shape[0], y.shape[0]
        total_pairs = Nx * Ny
        x_flat = x.unsqueeze(1).expand(Nx, Ny, -1).reshape(total_pairs, -1)
        y_flat = y.unsqueeze(0).expand(Nx, Ny, -1).reshape(total_pairs, -1)

        N = self.num_quad_steps
        t_grid = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype)
        dt = 1.0 / (N - 1)

        out_dists = torch.empty(total_pairs, device=x.device, dtype=x.dtype)

        # Process in chunks to prevent VRAM spikes when Nx * Ny is large (e.g. 256^2 = 65,536)
        for start in range(0, total_pairs, self.ot_chunk_size):
            end = min(start + self.ot_chunk_size, total_pairs)
            xb = x_flat[start:end]
            yb = y_flat[start:end]
            B_chunk = xb.shape[0]

            pts = [self.interpolant(xb, yb, tv.expand(B_chunk, 1)) for tv in t_grid]
            chunk_len = torch.zeros(B_chunk, device=x.device, dtype=x.dtype)

            for i in range(N - 1):
                mid = 0.5 * (pts[i] + pts[i + 1])
                vel = (pts[i + 1] - pts[i]) / dt
                speed = torch.sqrt(self.metric.norm_squared(mid, vel) + 1e-6)
                chunk_len = chunk_len + speed * dt

            out_dists[start:end] = chunk_len

        return out_dists.view(Nx, Ny)