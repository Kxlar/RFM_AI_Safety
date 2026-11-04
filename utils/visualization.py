import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from typing import Optional, Callable, Tuple
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.colors import LightSource


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


@torch.no_grad()
def compute_manifold_height(
    pts: torch.Tensor,
    metric: Optional[BaseRiemannianMetric] = None,
    potential: Optional[CompactForbiddenPotential] = None,
    include_obstacle: bool = True,
    obstacle_height_scale: float = 1.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a normalized 3D topographic height Z(x) in [0, 1+] representing the
    Riemannian metric cost landscape:
      - Base height Z_base(x) = normalized 0.5 * log det G(x) (low in data valleys, high off-manifold)
      - Obstacle spire Z_obs(x) = obstacle_height_scale * (V(x) / V_max)
    Returns (Z_total, Z_obs_only) as 1D numpy arrays of length [N].
    """
    if metric is not None:
        G = metric.metric_tensor(pts)  # [N, D, D]
        sign, logabsdet = torch.linalg.slogdet(G)
        z_base = 0.5 * logabsdet.detach().cpu().numpy()
        # Robust min-max normalization so valleys sit at 0.0 and off-manifold walls sit at 1.0
        p5, p95 = np.percentile(z_base, 5), np.percentile(z_base, 95)
        z_base = np.clip((z_base - p5) / (p95 - p5 + 1e-6), 0.0, 1.15)
    else:
        z_base = np.zeros(pts.shape[0], dtype=np.float32)

    z_obs = np.zeros_like(z_base)
    if include_obstacle and potential is not None:
        v_raw = potential.pointwise(pts).detach().cpu().numpy()
        v_norm = v_raw / ( float(potential.scale) + 1e-8 )
        z_obs = obstacle_height_scale * v_norm

    return z_base + z_obs, z_obs


@torch.no_grad()
def plot_3d_manifold_diagnostics(
    data: torch.Tensor,
    trajectories: torch.Tensor,
    forbidden_center: torch.Tensor,
    r_min: float,
    r_max: float,
    metric: Optional[BaseRiemannianMetric] = None,
    interpolant_net: Optional[MFMGeodesicInterpolantNet] = None,
    potential: Optional[CompactForbiddenPotential] = None,
    steered_field: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    title: str = "3D Riemannian Manifold & Obstacle Geodesics",
    output_path: str = "diagnostics_3d.png",
):
    """
    Generates a 3-panel 3D surface & trajectory figure:
      Panel 1: Base Riemannian Metric Landscape (0.5 * log det G(x)) showing the data manifold valleys.
      Panel 2: Manifold + Forbidden Energy Spire V(x) with 3D intrinsic boundary contours.
      Panel 3: Zoomed 3D Neighborhood around the Forbidden Zone showing Geodesics & Steered Flows
               routing around the obstacle spire along the manifold valley floor.
    """
    if not (output_path.endswith(".png") or output_path.endswith(".jpg") or output_path.endswith(".pdf")):
        output_path = output_path + ".png"
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = data.device
    data_np = data.detach().cpu().numpy()
    c_np = forbidden_center.detach().cpu().numpy().flatten()

    # 1. Global Grid for Panels 1 & 2
    res_global = 85
    gx, gy = np.meshgrid(np.linspace(-2.3, 2.3, res_global), np.linspace(-1.8, 2.0, res_global))
    grid_pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=-1), dtype=torch.float32, device=device)

    z_clean_flat, _ = compute_manifold_height(grid_pts, metric=metric, potential=potential, include_obstacle=False)
    z_total_flat, z_obs_flat = compute_manifold_height(grid_pts, metric=metric, potential=potential, include_obstacle=True)

    Z_clean = z_clean_flat.reshape(res_global, res_global)
    Z_total = z_total_flat.reshape(res_global, res_global)
    Z_obs = z_obs_flat.reshape(res_global, res_global)

    # Compute data point heights so we can scatter them inside the 3D valley floor
    sub_idx = np.random.choice(data.shape[0], size=min(1000, data.shape[0]), replace=False)
    data_sub = data[sub_idx]
    z_data_clean, _ = compute_manifold_height(data_sub, metric=metric, potential=potential, include_obstacle=False)
    z_data_total, _ = compute_manifold_height(data_sub, metric=metric, potential=potential, include_obstacle=True)
    data_sub_np = data_sub.detach().cpu().numpy()

    fig = plt.figure(figsize=(22, 7.2))
    ls = LightSource(azdeg=315, altdeg=45)

    # =========================================================================
    # PANEL 1: 3D Base Riemannian Manifold Landscape (Valleys = High Density)
    # =========================================================================
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    rgb_clean = ls.shade(Z_clean, cmap=cm.viridis, vert_exag=0.3, blend_mode="soft")
    ax0.plot_surface(
        gx, gy, Z_clean, facecolors=rgb_clean, rstride=1, cstride=1,
        antialiased=True, alpha=0.82, shade=False
    )
    ax0.scatter(
        data_sub_np[:, 0], data_sub_np[:, 1], z_data_clean + 0.03,
        c="white", edgecolors="black", linewidths=0.2, s=8, alpha=0.75, label="Data Manifold (Valley Floor)"
    )
    ax0.set_title("1. Base Riemannian Manifold Cost\nZ(x) = Norm(0.5 log det G(x))", fontsize=11.5, fontweight="bold")
    ax0.set_xlabel("x₁"); ax0.set_ylabel("x₂"); ax0.set_zlabel("Metric Cost Z(x)")
    ax0.set_xlim(-2.3, 2.3); ax0.set_ylim(-1.8, 2.0); ax0.set_zlim(0.0, 2.4)
    ax0.view_init(elev=38, azim=-62)
    ax0.legend(loc="upper left", fontsize=8.5)

    # =========================================================================
    # PANEL 2: 3D Manifold + Forbidden Area Potential Spire
    # =========================================================================
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    # Blend base viridis colormap with Crimson Red on the forbidden spire & clip strictly to [0, 1]
    base_rgba = cm.viridis(np.clip(Z_clean / 1.15, 0.0, 1.0))
    obs_weight = np.clip(Z_obs / 1.4, 0.0, 1.0)[..., None]
    spire_rgba = np.array([0.90, 0.12, 0.15, 1.0]).reshape(1, 1, 4)
    blended_rgba = np.clip((1.0 - obs_weight) * base_rgba + obs_weight * spire_rgba, 0.0, 1.0)

    ax1.plot_surface(
        gx, gy, Z_total, facecolors=blended_rgba, rstride=1, cstride=1,
        antialiased=True, alpha=0.85, shade=False
    )
    ax1.scatter(
        data_sub_np[:, 0], data_sub_np[:, 1], z_data_total + 0.03,
        c="white", edgecolors="black", linewidths=0.2, s=8, alpha=0.65
    )

    ax1.set_title("2. Manifold + Forbidden Barrier Spire\nZ(x) = Cost_G(x) + β V(x)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂"); ax1.set_zlabel("Total Energy Z(x)")
    ax1.set_xlim(-2.3, 2.3); ax1.set_ylim(-1.8, 2.0); ax1.set_zlim(0.0, 2.6)
    ax1.view_init(elev=38, azim=-62)

    legend_p2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=8, label="Forbidden Barrier Spire V(x)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="teal", markersize=8, label="Manifold Valley (Low G)"),
    ]
    ax1.legend(handles=legend_p2, loc="upper left", fontsize=8.5)

    # =========================================================================
    # PANEL 3: Zoomed 3D Neighborhood of Forbidden Zone + Geodesics & Flows
    # =========================================================================
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zoom_pad = max(r_max * 1.85, 1.15)
    x_min_z, x_max_z = c_np[0] - zoom_pad, c_np[0] + zoom_pad
    y_min_z, y_max_z = c_np[1] - zoom_pad, c_np[1] + zoom_pad * 0.75

    res_zoom = 75
    zx, zy = np.meshgrid(np.linspace(x_min_z, x_max_z, res_zoom), np.linspace(y_min_z, y_max_z, res_zoom))
    z_grid_pts = torch.tensor(np.stack([zx.ravel(), zy.ravel()], axis=-1), dtype=torch.float32, device=device)

    z_zoom_clean_f, _ = compute_manifold_height(z_grid_pts, metric=metric, potential=potential, include_obstacle=False)
    z_zoom_tot_f, z_zoom_obs_f = compute_manifold_height(z_grid_pts, metric=metric, potential=potential, include_obstacle=True)

    Z_zoom_clean = z_zoom_clean_f.reshape(res_zoom, res_zoom)
    Z_zoom_tot = z_zoom_tot_f.reshape(res_zoom, res_zoom)
    Z_zoom_obs = z_zoom_obs_f.reshape(res_zoom, res_zoom)

    zoom_base_rgba = cm.viridis(np.clip(Z_zoom_clean / 1.15, 0.0, 1.0))
    zoom_obs_w = np.clip(Z_zoom_obs / 1.4, 0.0, 1.0)[..., None]
    zoom_rgba = np.clip((1.0 - zoom_obs_w) * zoom_base_rgba + zoom_obs_w * spire_rgba, 0.0, 1.0)
    ax2.plot_surface(
        zx, zy, Z_zoom_tot, facecolors=zoom_rgba, rstride=1, cstride=1,
        antialiased=True, alpha=0.58, shade=False
    )

    # Generate Targeted Geodesics & Steered Trajectories in the Forbidden Neighborhood
    # Pick start points in an arc around the upper-left / top of the forbidden zone and targets on the right flank
    n_focus = 18
    angles_start = torch.linspace(0.65 * torch.pi, 1.35 * torch.pi, n_focus, device=device)
    rad_start = r_max * 1.35
    x_focus_start = forbidden_center.view(1, 2) + rad_start * torch.stack(
        [torch.cos(angles_start), torch.sin(angles_start)], dim=-1
    )

    # Target points on the opposite side of the forbidden peak along the top moon arch
    dists_to_c = torch.linalg.norm(data - forbidden_center.view(1, 2), dim=-1)
    right_flank_mask = (data[:, 0] > forbidden_center[0] + r_min * 0.9) & (data[:, 1] > 0.2) & (dists_to_c < zoom_pad * 1.2)
    right_flank_pts = data[right_flank_mask]
    if right_flank_pts.shape[0] >= n_focus:
        idx_t = torch.linspace(0, right_flank_pts.shape[0] - 1, n_focus, device=device).long()
        x_focus_end = right_flank_pts[idx_t]
    else:
        x_focus_end = forbidden_center.view(1, 2) + torch.tensor([[0.65, -0.35]], device=device).expand(n_focus, 2)

    # 3A. Plot 3D Geodesic Interpolants (Magenta Dashed) if MFM interpolant exists
    if interpolant_net is not None:
        t_steps = torch.linspace(0.0, 1.0, 45, device=device)
        geo_paths = torch.stack(
            [interpolant_net(x_focus_start, x_focus_end, tv.expand(n_focus, 1)) for tv in t_steps], dim=0
        )  # [T, B, 2]
        T_g, B_g, _ = geo_paths.shape
        z_geo_flat, _ = compute_manifold_height(
            geo_paths.view(T_g * B_g, 2), metric=metric, potential=potential, include_obstacle=True
        )
        z_geo = z_geo_flat.reshape(T_g, B_g) + 0.06  # Lift slightly above surface for visibility
        geo_np = geo_paths.detach().cpu().numpy()

        for b in range(B_g):
            ax2.plot(
                geo_np[:, b, 0], geo_np[:, b, 1], z_geo[:, b],
                color="magenta", linestyle="--", linewidth=1.8, alpha=0.95, zorder=10
            )

    # 3B. Plot Steered Flow Trajectories passing near the Forbidden Zone (Lime/Green Solid)
    # Filter trajectories from `trajectories` that pass within `zoom_pad` of `forbidden_center`
    traj_dists = torch.linalg.norm(trajectories - forbidden_center.view(1, 1, 2), dim=-1)
    near_mask = (traj_dists.min(dim=0)[0] < r_max * 1.35)
    near_trajs = trajectories[:, near_mask, :]
    if near_trajs.shape[1] == 0:
        near_trajs = trajectories[:, :min(20, trajectories.shape[1]), :]
    else:
        near_trajs = near_trajs[:, :min(22, near_trajs.shape[1]), :]

    T_tr, B_tr, _ = near_trajs.shape
    z_tr_flat, _ = compute_manifold_height(
        near_trajs.view(T_tr * B_tr, 2), metric=metric, potential=potential, include_obstacle=True
    )
    z_tr = z_tr_flat.reshape(T_tr, B_tr) + 0.05
    tr_np = near_trajs.detach().cpu().numpy()

    for b in range(B_tr):
        in_box = (
            (tr_np[:, b, 0] >= x_min_z) & (tr_np[:, b, 0] <= x_max_z) &
            (tr_np[:, b, 1] >= y_min_z) & (tr_np[:, b, 1] <= y_max_z)
        )
        if np.any(in_box):
            ax2.plot(
                tr_np[in_box, b, 0], tr_np[in_box, b, 1], z_tr[in_box, b],
                color="lime", linewidth=2.0, alpha=0.9, zorder=11
            )
            end_idx = np.where(in_box)[0][-1]
            ax2.scatter(
                tr_np[end_idx, b, 0], tr_np[end_idx, b, 1], z_tr[end_idx, b],
                color="darkgreen", edgecolors="white", s=24, zorder=12
            )

    ax2.set_title(f"3. Zoomed 3D Neighborhood & Geodesics\n{title}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂"); ax2.set_zlabel("Energy Z(x)")
    ax2.set_xlim(x_min_z, x_max_z); ax2.set_ylim(y_min_z, y_max_z); ax2.set_zlim(0.0, 2.6)
    ax2.view_init(elev=44, azim=-55)

    legend_p3 = [
        Line2D([0], [0], color="lime", lw=2.0, label="3D Steered Flow Trajectories"),
    ]
    if interpolant_net is not None:
        legend_p3.append(Line2D([0], [0], color="magenta", lw=1.8, ls="--", label="3D Geodesic Interpolants"))
    legend_p3.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=8, label="Forbidden Spire"))
    ax2.legend(handles=legend_p3, loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"3D Manifold diagnostics plot saved to {output_path}")