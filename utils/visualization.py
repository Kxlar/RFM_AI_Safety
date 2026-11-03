import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
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
    show_potential_heatmap: bool = True,
    title: str = "Riemannian Generative Flow Diagnostics",
    output_path: str = "diagnostics.png",
):
    if not (output_path.endswith(".png") or output_path.endswith(".jpg") or output_path.endswith(".pdf")):
        output_path = output_path + ".png"
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = data.device
    data_np = data.detach().cpu().numpy()
    traj_np = trajectories.detach().cpu().numpy()
    c_np = forbidden_center.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Precompute 2D evaluation grid for Intrinsic Geodesic Isocontours & Potential Heatmap
    grid_res = 90
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, grid_res), np.linspace(-2.0, 2.0, grid_res))
    grid_pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=-1), dtype=torch.float32, device=device)

    if potential is not None:
        dist_grid = potential.compute_distance(grid_pts).detach().cpu().numpy().reshape(grid_res, grid_res)
        pot_grid = potential.pointwise(grid_pts).detach().cpu().numpy().reshape(grid_res, grid_res)
    else:
        dist_grid = np.linalg.norm(
            grid_pts.detach().cpu().numpy() - c_np.reshape(1, 2), axis=-1
        ).reshape(grid_res, grid_res)
        pot_grid = None

    def draw_intrinsic_boundaries(ax, fill_core: bool = False):
        """Draws the exact intrinsic geodesic isocontours d_g(x, c) = r_min and d_g(x, c) = r_max."""
        if fill_core:
            ax.contourf(gx, gy, dist_grid, levels=[0.0, r_min], colors=["red"], alpha=0.35)
        ax.contour(gx, gy, dist_grid, levels=[r_min], colors=["red"], linewidths=2.0, linestyles="solid")
        ax.contour(gx, gy, dist_grid, levels=[r_max], colors=["darkorange"], linewidths=1.5, linestyles="dotted")
        ax.scatter(c_np[0], c_np[1], marker="+", color="darkred", s=45, zorder=5)

    # --- Panel 1: Metric Ellipses, Potential Heatmap & Geodesics ---
    ax0 = axes[0]
    if show_potential_heatmap and pot_grid is not None:
        ax0.contourf(gx, gy, pot_grid, levels=16, cmap="Reds", alpha=0.35)

    ax0.scatter(data_np[:, 0], data_np[:, 1], s=3, c="lightgray", alpha=0.5)

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
    has_geo = False
    if interpolant_net is not None:
        has_geo = True
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
            ax0.plot(geo_curves[:, b, 0], geo_curves[:, b, 1], "m--", lw=1.3, alpha=0.8)

    draw_intrinsic_boundaries(ax0, fill_core=not show_potential_heatmap)
    ax0.set_title("1. Metric Ellipses & Intrinsic Geodesic Isocontours", fontsize=12, fontweight="bold")
    ax0.set_xlim(-2.5, 2.5); ax0.set_ylim(-2.0, 2.0)

    legend_p1 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=5, label="Data Manifold"),
        Line2D([0], [0], color="red", lw=2.0, label="Intrinsic Core (d_g = r_min)"),
        Line2D([0], [0], color="darkorange", lw=1.5, ls=":", label="Geodesic Cutoff (d_g = r_max)"),
    ]
    if has_geo:
        legend_p1.append(Line2D([0], [0], color="m", lw=1.3, ls="--", label="Learned Geodesics"))
    ax0.legend(handles=legend_p1, loc="upper left", fontsize=8.5)

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

    draw_intrinsic_boundaries(ax1, fill_core=True)
    ax1.set_title("2. Steered Vector Field Streamlines (t = 0.5)", fontsize=12, fontweight="bold")
    ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.0, 2.0)
    ax1.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2.0, label="Intrinsic Core (d_g <= r_min)"),
            Line2D([0], [0], color="darkorange", lw=1.5, ls=":", label="Geodesic Cutoff (d_g = r_max)"),
        ],
        loc="upper left",
        fontsize=8.5,
    )

    # --- Panel 3: Simulated Particle Trajectories ---
    ax2 = axes[2]
    ax2.scatter(data_np[:, 0], data_np[:, 1], s=3, c="lightgray", alpha=0.5)
    draw_intrinsic_boundaries(ax2, fill_core=True)

    n_plot = min(65, traj_np.shape[1])
    for b in range(n_plot):
        lx, ly = traj_np[:, b, 0], traj_np[:, b, 1]
        ax2.plot(lx, ly, color="forestgreen", alpha=0.45, lw=1.2)
        ax2.scatter(lx[0], ly[0], marker="x", color="black", s=16, alpha=0.5)
        ax2.scatter(lx[-1], ly[-1], marker="o", color="darkgreen", s=18, alpha=0.9)

    ax2.set_title(f"3. {title}", fontsize=11, fontweight="bold")
    ax2.set_xlim(-2.5, 2.5); ax2.set_ylim(-2.0, 2.0)
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=5, label="Data Manifold"),
            Line2D([0], [0], color="red", lw=2.0, label="Intrinsic Core (d_g <= r_min)"),
            Line2D([0], [0], color="darkorange", lw=1.5, ls=":", label="Repulsion Cutoff (d_g = r_max)"),
            Line2D([0], [0], marker="x", color="w", markeredgecolor="black", markersize=6, label="Start (t=0)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="darkgreen", markersize=6, label="End (t=1)"),
        ],
        loc="upper left",
        fontsize=8.5,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagnostics plot saved to {output_path}")