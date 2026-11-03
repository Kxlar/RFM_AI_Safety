import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from typing import Optional, Callable

from geometry.metrics import BaseRiemannianMetric
from models.networks import MFMGeodesicInterpolantNet
from steering.repulsion import CompactForbiddenPotential


@torch.no_grad()
def plot_full_diagnostics(
    data: torch.Tensor,
    trajectories: torch.Tensor,
    forbidden_center: torch.Tensor,
    r_min: float,
    r_max: float,
    steered_field: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    metric: Optional[BaseRiemannianMetric] = None,
    interpolant_net: Optional[MFMGeodesicInterpolantNet] = None,
    potential: Optional[CompactForbiddenPotential] = None,
    title: str = "Riemannian Generative Flow Diagnostics",
    output_path: str = "diagnostics.png",
):
    if not output_path.endswith(".png") and not output_path.endswith(".jpg") and not output_path.endswith(".pdf"):
        output_path = output_path + ".png"
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = data.device
    data_np = data.detach().cpu().numpy()
    traj_np = trajectories.detach().cpu().numpy()
    c_np = forbidden_center.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # --- Panel 1: Metric Ellipses, Potential Heatmap & Geodesics ---
    ax0 = axes[0]
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 90), np.linspace(-2.0, 2.0, 90))
    grid_pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=-1), dtype=torch.float32, device=device)

    if potential is not None:
        pot_grid = potential.pointwise(grid_pts).detach().cpu().numpy().reshape(90, 90)
        ax0.contourf(gx, gy, pot_grid, levels=16, cmap="Reds", alpha=0.35)

    ax0.scatter(data_np[:, 0], data_np[:, 1], s=3, c="lightgray", alpha=0.5, label="Data Manifold")

    # Plot Tissot Indicatrices (Ellipses of co-metric Gamma = G^{-1})
    if metric is not None:
        ex, ey = np.meshgrid(np.linspace(-2.1, 2.1, 8), np.linspace(-1.6, 1.6, 8))
        e_pts = torch.tensor(np.stack([ex.ravel(), ey.ravel()], axis=-1), dtype=torch.float32, device=device)
        G_inv = metric.cometric_tensor(e_pts).detach().cpu().numpy()
        for i in range(e_pts.shape[0]):
            eigvals, eigvecs = np.linalg.eigh(G_inv[i])
            eigvals = np.clip(eigvals, 1e-4, 5.0)
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            w, h = 0.20 * np.sqrt(eigvals[1]), 0.20 * np.sqrt(eigvals[0])
            ell = Ellipse(
                xy=(e_pts[i, 0].item(), e_pts[i, 1].item()),
                width=w, height=h, angle=angle,
                edgecolor="teal", facecolor="none", alpha=0.55, linewidth=1.1,
            )
            ax0.add_patch(ell)

    # Plot MFM Geodesic Curves
    if interpolant_net is not None:
        n_geo = 16
        idx_start = torch.randint(0, data.shape[0], (n_geo,), device=device)
        idx_end = torch.randint(0, data.shape[0], (n_geo,), device=device)
        x_s = data[idx_start]
        x_e = data[idx_end]
        t_vals = torch.linspace(0.0, 1.0, 50, device=device)
        geo_curves = torch.stack(
            [interpolant_net(x_s, x_e, tv.expand(n_geo, 1)) for tv in t_vals], dim=0
        ).cpu().numpy()
        for b in range(n_geo):
            ax0.plot(
                geo_curves[:, b, 0], geo_curves[:, b, 1],
                "m--", lw=1.3, alpha=0.8, label="Learned Geodesics" if b == 0 else ""
            )

    ax0.add_patch(Circle((c_np[0], c_np[1]), r_min, color="red", fill=False, lw=2.0, label="Forbidden Core (r_min)"))
    ax0.add_patch(Circle((c_np[0], c_np[1]), r_max, color="darkorange", fill=False, lw=1.5, ls=":", label="Geodesic Cutoff (r_max)"))
    ax0.set_title("1. Metric Ellipses & Geodesic Interpolants", fontsize=12, fontweight="bold")
    ax0.set_xlim(-2.5, 2.5); ax0.set_ylim(-2.0, 2.0)
    ax0.legend(loc="upper left", fontsize=8.5)

    # --- Panel 2: Vector Field Streamlines at t = 0.5 ---
    ax1 = axes[1]
    ax1.scatter(data_np[:, 0], data_np[:, 1], s=3, c="lightgray", alpha=0.45)
    if steered_field is not None:
        sx, sy = np.meshgrid(np.linspace(-2.5, 2.5, 30), np.linspace(-2.0, 2.0, 30))
        s_pts = torch.tensor(np.stack([sx.ravel(), sy.ravel()], axis=-1), dtype=torch.float32, device=device)
        v_mid = steered_field(s_pts, 0.5).detach().cpu().numpy()
        U = v_mid[:, 0].reshape(30, 30)
        V = v_mid[:, 1].reshape(30, 30)
        speed = np.sqrt(U ** 2 + V ** 2)
        ax1.streamplot(sx, sy, U, V, color=speed, cmap="viridis", density=1.2, linewidth=1.0)

    ax1.add_patch(Circle((c_np[0], c_np[1]), r_min, color="red", alpha=0.4, label="Forbidden Core"))
    ax1.add_patch(Circle((c_np[0], c_np[1]), r_max, color="darkorange", fill=False, lw=1.5, ls=":", label="Geodesic Cutoff"))
    ax1.set_title("2. Steered Vector Field Streamlines (t = 0.5)", fontsize=12, fontweight="bold")
    ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.0, 2.0)
    ax1.legend(loc="upper left", fontsize=8.5)

    # --- Panel 3: Simulated Particle Trajectories ---
    ax2 = axes[2]
    ax2.scatter(data_np[:, 0], data_np[:, 1], s=3, c="lightgray", alpha=0.5, label="Data Manifold")
    ax2.add_patch(Circle((c_np[0], c_np[1]), r_min, color="red", alpha=0.35, label="Forbidden Core"))
    ax2.add_patch(Circle((c_np[0], c_np[1]), r_max, color="darkorange", fill=False, lw=1.5, ls=":", label="Repulsion Cutoff"))

    n_plot = min(65, traj_np.shape[1])
    for b in range(n_plot):
        lx, ly = traj_np[:, b, 0], traj_np[:, b, 1]
        ax2.plot(lx, ly, color="forestgreen", alpha=0.45, lw=1.2)
        ax2.scatter(lx[0], ly[0], marker="x", color="black", s=16, alpha=0.5, label="Start (t=0)" if b == 0 else "")
        ax2.scatter(lx[-1], ly[-1], marker="o", color="darkgreen", s=18, alpha=0.9, label="End (t=1)" if b == 0 else "")

    ax2.set_title(f"3. {title}", fontsize=12, fontweight="bold")
    ax2.set_xlim(-2.5, 2.5); ax2.set_ylim(-2.0, 2.0)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagnostics plot saved to {output_path}")