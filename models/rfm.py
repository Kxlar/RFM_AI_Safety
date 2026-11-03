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
        x_clean = x.detach().clone().requires_grad_(True)
        x1_clean = x1.detach().clone()

        with torch.enable_grad():
            d_val = self.premetric.distance(x_clean, x1_clean)  # [B]
            grad_euc = torch.autograd.grad(
                outputs=d_val.sum(),
                inputs=x_clean,
                create_graph=False,
                retain_graph=False,
            )[0]

        # Immediately detach both d_val and grad_euc so zero graph references survive
        d_val_det = d_val.detach()
        grad_euc_det = grad_euc.detach()
        x_det = x.detach()
        t_det = t.detach()

        with torch.no_grad():
            if self.metric is not None:
                grad_g = self.metric.riemannian_gradient(x_det, grad_euc_det)
                norm_g = torch.sqrt(
                    torch.clamp(self.metric.norm_squared(x_det, grad_g), min=1e-8)
                ).unsqueeze(-1)
                unit_dir = grad_g / norm_g
            else:
                norm_euc = torch.linalg.norm(grad_euc_det, dim=-1, keepdim=True) + 1e-8
                unit_dir = grad_euc_det / norm_euc

            denom = torch.clamp(1.0 - t_det, min=0.04)
            ut = -(d_val_det.unsqueeze(-1) / denom) * unit_dir

        return ut.detach()

    def sample_conditional_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0_d = x0.detach()
        x1_d = x1.detach()
        t_d = t.detach()

        if self.premetric is None:
            xt = (1.0 - t_d) * x0_d + t_d * x1_d
            ut = x1_d - x0_d
            return xt.detach(), ut.detach()

        if not self.simulate_premetric_ode:
            xt = (1.0 - t_d) * x0_d + t_d * x1_d
            ut = self._premetric_conditional_velocity(xt, x1_d, t_d)
            return xt.detach(), ut.detach()

        xt = x0_d.clone()
        dt_step = t_d / self.ode_sim_steps
        s = torch.zeros_like(t_d)
        for _ in range(self.ode_sim_steps):
            u_s = self._premetric_conditional_velocity(xt, x1_d, s)
            xt = (xt + dt_step * u_s).detach()
            s = s + dt_step

        ut = self._premetric_conditional_velocity(xt, x1_d, t_d)
        return xt.detach(), ut.detach()

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
            torch.utils.data.TensorDataset(data.detach()), batch_size=batch_size, shuffle=True
        )

        pbar = tqdm(range(epochs), desc="Training RFM")
        for _ in pbar:
            epoch_loss = 0.0
            for (x1,) in loader:
                x1 = x1.to(device).detach()
                x0 = torch.randn_like(x1)
                if self.coupler is not None:
                    with torch.no_grad():
                        x0, x1 = self.coupler.couple(x0, x1)
                        x0, x1 = x0.detach(), x1.detach()

                t = (torch.rand(x1.shape[0], 1, device=device) * 0.96).detach()
                xt, ut = self.sample_conditional_path(x0, x1, t)

                # Strictly verify xt, ut, t have no autograd history before forward pass
                xt = xt.detach()
                ut = ut.detach()
                t = t.detach()

                vt = self.v_net(xt, t)
                diff = vt - ut

                if self.metric is not None:
                    with torch.no_grad():
                        G = self.metric.metric_tensor(xt).detach()
                    loss = torch.mean(torch.einsum("bi,bij,bj->b", diff, G, diff))
                else:
                    loss = torch.mean(diff ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            pbar.set_postfix(rcfm_loss=f"{epoch_loss / len(loader):.4f}")