import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Tuple

from models.networks import TimeVelocityMLP
from geometry.metrics import BaseRiemannianMetric
from geometry.premetrics import BasePremetric
from transport.ot_couplings import BaseCoupler


class RiemannianFlowMatching(nn.Module):
    """
    Riemannian Flow Matching (Chen & Lipman, 2023).
    Supports both premetric-driven ODE conditional trajectories (for Biharmonic, Diffusion,
    or Geodesic premetrics) and metric-weighted RCFM loss under G(x).
    """
    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 256,
        metric: Optional[BaseRiemannianMetric] = None,
        premetric: Optional[BasePremetric] = None,
        coupler: Optional[BaseCoupler] = None,
        simulate_premetric_ode: bool = False,
        ode_sim_steps: int = 20,
    ):
        super().__init__()
        self.v_net = TimeVelocityMLP(dim=dim, hidden_dim=hidden_dim)
        self.metric = metric
        self.premetric = premetric
        self.coupler = coupler
        self.simulate_premetric_ode = simulate_premetric_ode
        self.ode_sim_steps = ode_sim_steps

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.v_net(x, t)

    def _premetric_conditional_velocity(
        self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the RFM conditional vector field from a premetric d(x, x1):
            u_t(x | x1) = - [d(x, x1) / (1 - t)] * grad_g d(x, x1) / ||grad_g d(x, x1)||_g
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            d_val = self.premetric.distance(x_in, x1)  # [B]
            grad_euc = torch.autograd.grad(d_val.sum(), x_in)[0]  # [B, D]

        if self.metric is not None:
            grad_g = self.metric.riemannian_gradient(x, grad_euc)
            norm_g = torch.sqrt(torch.clamp(self.metric.norm_squared(x, grad_g), min=1e-8)).unsqueeze(-1)
            unit_dir = grad_g / norm_g
        else:
            norm_euc = torch.linalg.norm(grad_euc, dim=-1, keepdim=True) + 1e-8
            unit_dir = grad_euc / norm_euc

        denom = torch.clamp(1.0 - t, min=0.04)
        return -(d_val.unsqueeze(-1) / denom) * unit_dir

    def sample_conditional_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.premetric is None:
            xt = (1.0 - t) * x0 + t * x1
            ut = x1 - x0
            return xt, ut

        if not self.simulate_premetric_ode:
            xt = (1.0 - t) * x0 + t * x1
            ut = self._premetric_conditional_velocity(xt, x1, t)
            return xt, ut

        xt = x0.clone()
        dt_step = t / self.ode_sim_steps
        s = torch.zeros_like(t)
        for _ in range(self.ode_sim_steps):
            u_s = self._premetric_conditional_velocity(xt, x1, s)
            xt = (xt + dt_step * u_s).detach()
            s = s + dt_step

        ut = self._premetric_conditional_velocity(xt, x1, t)
        return xt, ut

    def fit(
        self,
        data: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.to(device)
        optimizer = optim.AdamW(self.v_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
        )

        pbar = tqdm(range(epochs), desc="Training RFM")
        for _ in pbar:
            epoch_loss = 0.0
            for (x1,) in loader:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                if self.coupler is not None:
                    x0, x1 = self.coupler.couple(x0, x1)

                t = torch.rand(x1.shape[0], 1, device=device) * 0.96
                xt, ut = self.sample_conditional_path(x0, x1, t)
                vt = self.v_net(xt, t)
                diff = vt - ut

                if self.metric is not None:
                    loss = torch.mean(self.metric.norm_squared(xt, diff))
                else:
                    loss = torch.mean(diff ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            pbar.set_postfix(rcfm_loss=f"{epoch_loss / len(loader):.4f}")