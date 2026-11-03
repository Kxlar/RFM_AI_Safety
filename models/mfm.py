import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional

from models.networks import TimeVelocityMLP, MFMGeodesicInterpolantNet
from geometry.metrics import BaseRiemannianMetric
from transport.ot_couplings import BaseCoupler


class MetricFlowMatching(nn.Module):
    """
    Metric Flow Matching (Kapusniak et al., 2024) with Potential V_t(x) Action Minimization.
      Stage 1: Optimize MFMGeodesicInterpolantNet gamma_eta to minimize Kinetic + Potential Energy:
               L(eta) = E_{x0, x1, t} [ 1/2 || dot_gamma_eta(t) ||_{G(gamma_eta(t))}^2 + V_t(gamma_eta(t)) ]
      Stage 2: Train TimeVelocityMLP v_theta(x, t) on the induced geodesic vector field dot_gamma_eta(t).
               At test time, only v_theta is integrated (metric G is dropped).
    """
    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 256,
        metric: Optional[BaseRiemannianMetric] = None,
        potential: Optional[nn.Module] = None,
        coupler: Optional[BaseCoupler] = None,
    ):
        super().__init__()
        self.interpolant_net = MFMGeodesicInterpolantNet(dim=dim, hidden_dim=hidden_dim)
        self.v_net = TimeVelocityMLP(dim=dim, hidden_dim=hidden_dim)
        self.metric = metric
        self.potential = potential
        self.coupler = coupler

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.v_net(x, t)

    def fit_interpolant(
        self,
        data: torch.Tensor,
        epochs: int = 600,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """Stage 1: Train the neural geodesic interpolant gamma_eta(x0, x1, t)."""
        self.to(device)
        for p in self.interpolant_net.parameters():
            p.requires_grad_(True)

        optimizer = optim.AdamW(self.interpolant_net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
        )

        pbar = tqdm(range(epochs), desc="MFM Stage 1 (Geodesic + V_t Action)")
        for _ in pbar:
            total_action = 0.0
            for (x1,) in loader:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                if self.coupler is not None:
                    x0, x1 = self.coupler.couple(x0, x1)

                t = torch.rand(x1.shape[0], 1, device=device)
                xt, dot_xt = self.interpolant_net.position_and_velocity(x0, x1, t)

                if self.metric is not None:
                    G = self.metric.metric_tensor(xt)
                    kinetic = 0.5 * torch.bmm(
                        dot_xt.unsqueeze(1), torch.bmm(G, dot_xt.unsqueeze(-1))
                    ).squeeze(-1).squeeze(-1)
                else:
                    kinetic = 0.5 * torch.sum(dot_xt ** 2, dim=-1)

                if self.potential is not None:
                    pot_val = self.potential(xt, t)
                else:
                    pot_val = 0.0

                loss = torch.mean(kinetic + pot_val)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.interpolant_net.parameters(), 1.0)
                optimizer.step()
                total_action += loss.item()

            scheduler.step()
            pbar.set_postfix(action=f"{total_action / len(loader):.4f}")

    def fit_flow(
        self,
        data: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """Stage 2: Freeze gamma_eta and train v_theta(x, t) to match dot_gamma_eta."""
        self.to(device)
        for p in self.interpolant_net.parameters():
            p.requires_grad_(False)

        optimizer = optim.AdamW(self.v_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
        )

        pbar = tqdm(range(epochs), desc="MFM Stage 2 (Flow Matching)")
        for _ in pbar:
            total_loss = 0.0
            for (x1,) in loader:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                if self.coupler is not None:
                    x0, x1 = self.coupler.couple(x0, x1)

                t = torch.rand(x1.shape[0], 1, device=device)
                xt, dot_xt = self.interpolant_net.position_and_velocity(x0, x1, t)
                xt = xt.detach()
                ut = dot_xt.detach()

                vt = self.v_net(xt, t)
                diff = vt - ut

                if self.metric is not None:
                    with torch.no_grad():
                        G = self.metric.metric_tensor(xt).detach()
                    loss = torch.mean(torch.bmm(diff.unsqueeze(1), torch.bmm(G, diff.unsqueeze(-1))))
                else:
                    loss = torch.mean(diff ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            pbar.set_postfix(mfm_loss=f"{total_loss / len(loader):.4f}")

    def fit(
        self,
        data: torch.Tensor,
        epochs_interpolant: int = 600,
        epochs_flow: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.fit_interpolant(data, epochs=epochs_interpolant, batch_size=batch_size, lr=lr, device=device)
        self.fit_flow(data, epochs=epochs_flow, batch_size=batch_size, lr=lr, device=device)