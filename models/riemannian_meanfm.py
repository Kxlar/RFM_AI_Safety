import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jvp
from tqdm import tqdm
from typing import Optional, Tuple

from models.networks import IntervalVelocityMLP, MFMGeodesicInterpolantNet
from geometry.metrics import BaseRiemannianMetric
from transport.ot_couplings import BaseCoupler


class RiemannianMeanFlow(nn.Module):
    """
    Riemannian MeanFlow (adapted from Geng et al., 2025 to curved geodesic manifolds).
    Learns the average velocity u_theta(x_t, r, t) along geodesic interpolants gamma_eta(t)
    using forward-mode JVP:
        u_target = stopgrad( v_t - (t - r) * (d_x u_theta * v_t + d_t u_theta) )
    """
    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 256,
        interpolant_net: Optional[MFMGeodesicInterpolantNet] = None,
        metric: Optional[BaseRiemannianMetric] = None,
        coupler: Optional[BaseCoupler] = None,
    ):
        super().__init__()
        self.u_net = IntervalVelocityMLP(dim=dim, hidden_dim=hidden_dim)
        self.interpolant_net = interpolant_net
        self.metric = metric
        self.coupler = coupler

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous velocity field v_t(x) = u_theta(x, t, t) when interval r -> t."""
        return self.u_net(x, t, t)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples in `steps` finite jumps using the learned average velocity u_theta(x_r, r, t).
        Setting steps=1 performs single-step generation: x_1 = x_0 + u_theta(x_0, 0, 1).
        """
        x = x0.clone()
        B = x0.shape[0]
        t_grid = torch.linspace(0.0, 1.0, steps + 1, device=x0.device, dtype=x0.dtype)
        traj = [x.clone()]

        for i in range(steps):
            r_t = t_grid[i].expand(B, 1)
            t_t = t_grid[i + 1].expand(B, 1)
            dt = t_t - r_t
            x = x + dt * self.u_net(x, r_t, t_t)
            traj.append(x.clone())

        return x, torch.stack(traj, dim=0)

    def fit(
        self,
        data: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.to(device)
        if self.interpolant_net is not None:
            for p in self.interpolant_net.parameters():
                p.requires_grad_(False)

        optimizer = optim.AdamW(self.u_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
        )

        pbar = tqdm(range(epochs), desc="Training Riemannian MeanFlow")
        for _ in pbar:
            total_loss = 0.0
            for (x1,) in loader:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                if self.coupler is not None:
                    x0, x1 = self.coupler.couple(x0, x1)

                B = x1.shape[0]
                t_a = torch.rand(B, 1, device=device)
                t_b = torch.rand(B, 1, device=device)
                r = torch.minimum(t_a, t_b)
                t = torch.maximum(t_a, t_b)

                if self.interpolant_net is not None:
                    xt, vt = self.interpolant_net.position_and_velocity(x0, x1, t)
                    xt, vt = xt.detach(), vt.detach()
                else:
                    xt = (1.0 - t) * x0 + t * x1
                    vt = x1 - x0

                def forward_u(x_in, r_in, t_in):
                    return self.u_net(x_in, r_in, t_in)

                dr = torch.zeros_like(r)
                dt = torch.ones_like(t)
                u_pred, du_dt = jvp(forward_u, (xt, r, t), (vt, dr, dt))

                u_target = (vt - (t - r) * du_dt).detach()
                diff = u_pred - u_target

                if self.metric is not None:
                    G = self.metric.metric_tensor(xt)
                    loss = torch.mean(torch.bmm(diff.unsqueeze(1), torch.bmm(G, diff.unsqueeze(-1))))
                else:
                    loss = torch.mean(diff ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.u_net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            pbar.set_postfix(rmean_loss=f"{total_loss / len(loader):.4f}")