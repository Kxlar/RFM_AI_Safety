import torch
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment

from geometry.premetrics import BasePremetric


class BaseCoupler:
    """Base interface for pairing prior samples x0 with target data samples x1."""
    def couple(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class IndependentCoupler(BaseCoupler):
    """Standard independent coupling: (x0, x1) ~ p0(x0) p1(x1)."""
    def couple(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x0, x1


class MinibatchOTCoupler(BaseCoupler):
    """
    Minibatch Optimal Transport coupler supporting Euclidean or Premetric/Geodesic
    cost matrices C_{ij} = d(x0_i, x1_j)^2 via exact Hungarian assignment or Sinkhorn.
    """
    def __init__(
        self,
        premetric: Optional[BasePremetric] = None,
        method: str = "exact",
        sinkhorn_reg: float = 0.05,
        sinkhorn_iters: int = 50,
    ):
        """
        Args:
            premetric: Instance of BasePremetric (EuclideanPremetric, DiffusionPremetric,
                       BiharmonicPremetric, or MFMGeodesicPremetric). If None, uses squared Euclidean.
            method: 'exact' (Hungarian algorithm) or 'sinkhorn'.
            sinkhorn_reg: Entropic regularization epsilon for Sinkhorn.
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations.
        """
        self.premetric = premetric
        self.method = method
        self.reg = sinkhorn_reg
        self.iters = sinkhorn_iters

    @torch.no_grad()
    def compute_cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if self.premetric is not None:
            if hasattr(self.premetric, "pairwise_distance"):
                dists = self.premetric.pairwise_distance(x0, x1)
            else:
                B = x0.shape[0]
                x0_rep = x0.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
                x1_rep = x1.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
                dists = self.premetric.distance(x0_rep, x1_rep).view(B, B)
            return dists ** 2
        return torch.cdist(x0, x1, p=2.0) ** 2

    @torch.no_grad()
    def couple(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x0.shape[0]
        C = self.compute_cost_matrix(x0, x1)

        if self.method == "exact":
            C_np = C.detach().cpu().numpy()
            _, col_ind = linear_sum_assignment(C_np)
            idx = torch.tensor(col_ind, dtype=torch.long, device=x1.device)
            return x0, x1[idx]

        elif self.method == "sinkhorn":
            C_scaled = C / (C.max() + 1e-8)
            K = torch.exp(-C_scaled / self.reg)
            u = torch.full((B,), 1.0 / B, device=x0.device, dtype=x0.dtype)
            v = torch.full((B,), 1.0 / B, device=x0.device, dtype=x0.dtype)

            for _ in range(self.iters):
                u = 1.0 / (B * (K @ v + 1e-12))
                v = 1.0 / (B * (K.T @ u + 1e-12))

            P = u.unsqueeze(1) * K * v.unsqueeze(0)
            idx = torch.multinomial(P + 1e-12, num_samples=1).squeeze(-1)
            return x0, x1[idx]

        else:
            raise ValueError(f"Unsupported OT method: {self.method}")