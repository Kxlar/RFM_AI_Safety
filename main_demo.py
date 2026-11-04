import os
import csv
import time
import argparse
import torch
from sklearn.datasets import make_moons

from geometry.metrics import (
    CDCMetric,
    LANDMetric,
    RBFMetric,
    ConformalObstacleMetric,
)
from geometry.premetrics import (
    EuclideanPremetric,
    SpectralLaplacianPremetric,
    MFMGeodesicPremetric,
)
from transport.ot_couplings import IndependentCoupler, MinibatchOTCoupler
from models.rfm import RiemannianFlowMatching
from models.mfm import MetricFlowMatching
from models.riemannian_meanfm import RiemannianMeanFlow
from steering.repulsion import CompactForbiddenPotential, RiemannianRepulsionStreamer, integrate_ode
from utils.visualization import plot_full_diagnostics, plot_3d_manifold_diagnostics


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.25, 0.5, 1.0)) -> float:
    """Computes multi-scale Gaussian RBF Maximum Mean Discrepancy (MMD)."""
    with torch.no_grad():
        dist_xx = torch.cdist(x, x, p=2.0) ** 2
        dist_yy = torch.cdist(y, y, p=2.0) ** 2
        dist_xy = torch.cdist(x, y, p=2.0) ** 2

        k_xx, k_yy, k_xy = 0.0, 0.0, 0.0
        for s in sigmas:
            gamma = 1.0 / (2.0 * s * s)
            k_xx = k_xx + torch.exp(-gamma * dist_xx)
            k_yy = k_yy + torch.exp(-gamma * dist_yy)
            k_xy = k_xy + torch.exp(-gamma * dist_xy)

        mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        return max(mmd.item(), 0.0)


@torch.no_grad()
def compute_oracle_mmd_baselines(safe_gt_data: torch.Tensor, n_eval_samples: int, n_trials: int = 10) -> tuple:
    """
    Computes two ground-truth Oracle MMD baselines averaged over `n_trials` random splits:
      1. Half-Dataset Split MMD: MMD between two disjoint random halves of `safe_gt_data`.
      2. Sample-Size Matched MMD: MMD between `n_eval_samples` random safe points and `safe_gt_data`.
    """
    N = safe_gt_data.shape[0]
    half_mmds, sample_mmds = [], []
    for _ in range(n_trials):
        perm = torch.randperm(N, device=safe_gt_data.device)
        h1 = safe_gt_data[perm[: N // 2]]
        h2 = safe_gt_data[perm[N // 2 : 2 * (N // 2)]]
        half_mmds.append(compute_mmd_rbf(h1, h2))

        sub = safe_gt_data[perm[: min(n_eval_samples, N)]]
        sample_mmds.append(compute_mmd_rbf(sub, safe_gt_data))

    return sum(half_mmds) / len(half_mmds), sum(sample_mmds) / len(sample_mmds)


def get_top_moon_peak(raw_y: torch.Tensor, data_std_x: torch.Tensor) -> torch.Tensor:
    """Finds the standardized coordinates of the crest/peak of the upper half-moon (label == 0)."""
    top_moon = data_std_x[raw_y == 0]
    top_apex_mask = top_moon[:, 1] >= torch.quantile(top_moon[:, 1], 0.92)
    return top_moon[top_apex_mask].mean(dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Modular Riemannian Generative Flows Benchmark")
    parser.add_argument("--model", type=str, default="mfm", choices=["rfm", "mfm", "rmeanfm"])
    parser.add_argument("--metric", type=str, default="cdc", choices=["cdc", "land", "rbf", "none"])
    parser.add_argument(
        "--premetric",
        type=str,
        default="geodesic",
        choices=["euclidean", "diffusion", "biharmonic", "geodesic", "none"],
    )
    parser.add_argument("--ot", type=str, default="premetric", choices=["none", "euclidean", "premetric"])
    parser.add_argument("--ot-method", type=str, default="exact", choices=["exact", "sinkhorn"])
    parser.add_argument(
        "--obstacle-mode",
        type=str,
        default="both",
        choices=["none", "baked", "inference", "both"],
        help="'baked': bake V_t(x) into MFM geodesic action / Conformal Metric; 'inference': geodesic-scaled repulsion",
    )
    parser.add_argument("--plot-3d", action="store_true", default=True, help="Also save a 3D manifold & geodesic surface plot")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--epochs-cdc", type=int, default=800)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=1.2, help="Inference repulsion strength")
    parser.add_argument("--gyro-curl", type=float, default=0.6, help="Gyroscopic circulation to slide around obstacles")
    parser.add_argument("--baked-scale", type=float, default=45.0, help="Potential V(x) weight during MFM Stage 1 / Conformal baking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (auto-named if omitted)")
    parser.add_argument("--csv-path", type=str, default="./results_sweep/summary_results.csv", help="CSV file to log results")
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    t_start = time.perf_counter()

    # Auto-generate descriptive filename if --output is not specified
    ot_str = "ot_none" if args.ot == "none" else f"ot_{args.ot}_{args.ot_method}"
    is_euc_base = args.metric == "none" and args.premetric == "euclidean"
    prefix = "BASELINE_EUC_" if is_euc_base else ""
    auto_fname = f"{prefix}{args.model}_{args.metric}_{args.premetric}_{ot_str}_{args.obstacle_mode}.png"

    if args.output is None:
        out_dir = os.path.dirname(args.csv_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, auto_fname)
    else:
        out_path = args.output if args.output.endswith(".png") else args.output + ".png"

    print("=" * 75)
    print(f"Model: {args.model.upper()} | Metric: {args.metric.upper()} | Premetric: {args.premetric.upper()}")
    print(f"OT: {args.ot.upper()} ({args.ot_method}) | Obstacle Mode: {args.obstacle_mode.upper()} | Device: {dev}")
    print(f"Target Output: {out_path}")
    print("=" * 75)

    # 1. Generate Standardized 2-Moons Dataset & Peak Forbidden Zone
    torch.manual_seed(42)
    raw_x, raw_y = make_moons(n_samples=4000, noise=0.07, random_state=42)
    raw_x_t = torch.tensor(raw_x, dtype=torch.float32, device=dev)
    raw_y_t = torch.tensor(raw_y, dtype=torch.long, device=dev)
    data = (raw_x_t - raw_x_t.mean(dim=0)) / raw_x_t.std(dim=0)

    forbid_center = get_top_moon_peak(raw_y_t, data).to(dev)
    r_min, r_max = 0.32, 0.70

    # Reference safe ground-truth data & Oracle MMD Baselines
    dists_gt = torch.linalg.norm(data - forbid_center.unsqueeze(0), dim=-1)
    safe_gt_data = data[dists_gt >= r_min]
    oracle_half_mmd, oracle_sample_mmd = compute_oracle_mmd_baselines(safe_gt_data, args.n_samples)
    print(f"[Oracle Benchmark] Half-Dataset Split MMD: {oracle_half_mmd:.5f} | {args.n_samples}-Sample Oracle MMD: {oracle_sample_mmd:.5f}")

    # 2. Initialize & Fit Base Riemannian Metric G(x) (with CDC Caching)
    base_metric = None
    if args.metric == "cdc":
        base_metric = CDCMetric(dim=2, rank=2, hidden_dim=128, lambda_reg=1e-2).to(dev)
        cdc_cache = os.path.join(os.path.dirname(args.csv_path) or ".", "cached_cdc_metric.pt")
        if os.path.exists(cdc_cache):
            print(f"Loading cached CDC Metric from {cdc_cache}...")
            base_metric.load_state_dict(torch.load(cdc_cache, map_location=dev))
            base_metric.eval()
        else:
            print("Training CDC Metric Matching Network...")
            base_metric.fit(data, epochs=args.epochs_cdc, batch_size=256, lr=1e-3, device=dev)
            os.makedirs(os.path.dirname(cdc_cache) or ".", exist_ok=True)
            torch.save(base_metric.state_dict(), cdc_cache)
    elif args.metric == "land":
        base_metric = LANDMetric(data=data, max_Reference_points=600, sigma=0.25, lambda_reg=1e-2).to(dev)
    elif args.metric == "rbf":
        base_metric = RBFMetric(data=data, num_centers=80, bandwidth=None, lambda_reg=1e-2).to(dev)

    # 3. Initialize Premetric d(x, y)
    premetric = None
    if args.premetric == "euclidean":
        premetric = EuclideanPremetric().to(dev)
    elif args.premetric in ["diffusion", "biharmonic"]:
        premetric = SpectralLaplacianPremetric(
            data=data, mode=args.premetric, num_landmarks=512, num_eigen=32, sigma=0.35
        ).to(dev)

    # 4. Forbidden Region & Calibrated Potential V_t(x)
    # Note: CompactForbiddenPotential automatically calibrates `premetric` so r_min/r_max match physical radii!
    potential = CompactForbiddenPotential(
        center=forbid_center,
        r_min=r_min,
        r_max=r_max,
        scale=args.baked_scale,
        premetric=premetric,
    ).to(dev)

    is_baked = args.obstacle_mode in ["baked", "both"]
    active_metric = base_metric
    if is_baked and base_metric is not None and args.model == "rfm":
        active_metric = ConformalObstacleMetric(base_metric=base_metric, potential_fn=potential, beta=1.0).to(dev)

    # 5. Configure Optimal Transport Coupler
    if args.ot == "none":
        coupler = IndependentCoupler()
    elif args.ot == "euclidean":
        coupler = MinibatchOTCoupler(premetric=EuclideanPremetric().to(dev), method=args.ot_method)
    elif args.ot == "premetric":
        coupler = MinibatchOTCoupler(premetric=premetric, method=args.ot_method)

    # 6. Train Selected Generative Flow Model
    interpolant_net = None

    if args.model == "rfm":
        model = RiemannianFlowMatching(
            dim=2,
            metric=active_metric,
            premetric=premetric,
            coupler=coupler,
            simulate_premetric_ode=False,
        ).to(dev)
        model.fit(data, epochs=args.epochs, device=dev)

    elif args.model == "mfm":
        baked_pot = potential if is_baked else None
        model = MetricFlowMatching(
            dim=2,
            metric=active_metric,
            potential=baked_pot,
            coupler=coupler,
        ).to(dev)
        # Stage 1: Minimize Kinetic Energy + Forbidden Potential V_t(gamma_eta(t))
        model.fit_interpolant(data, epochs=int(args.epochs * 0.75), device=dev)
        interpolant_net = model.interpolant_net

        # Wrap the trained MFM Geodesic Interpolant as a fast amortized Geodesic Premetric (num_quad_steps=6 for fast OT)
        if base_metric is not None:
            geo_premetric = MFMGeodesicPremetric(
                mfm_interpolant=interpolant_net, metric=base_metric, num_quad_steps=6
            ).to(dev)
            if args.premetric == "geodesic":
                potential.premetric = geo_premetric  # Automatically recalibrates distance scale!
                if args.ot == "premetric":
                    model.coupler = MinibatchOTCoupler(premetric=geo_premetric, method=args.ot_method)

        # Stage 2: Train Velocity Field v_theta (drops metric at inference time!)
        model.fit_flow(data, epochs=args.epochs, device=dev)

    elif args.model == "rmeanfm":
        baked_pot = potential if is_baked else None
        mfm_stage1 = MetricFlowMatching(
            dim=2, metric=active_metric, potential=baked_pot, coupler=coupler
        ).to(dev)
        mfm_stage1.fit_interpolant(data, epochs=int(args.epochs * 0.65), device=dev)
        interpolant_net = mfm_stage1.interpolant_net

        if base_metric is not None and args.premetric == "geodesic":
            geo_premetric = MFMGeodesicPremetric(
                mfm_interpolant=interpolant_net, metric=base_metric, num_quad_steps=6
            ).to(dev)
            potential.premetric = geo_premetric
            if args.ot == "premetric":
                coupler = MinibatchOTCoupler(premetric=geo_premetric, method=args.ot_method)

        model = RiemannianMeanFlow(
            dim=2,
            interpolant_net=interpolant_net,
            metric=active_metric,
            coupler=coupler,
        ).to(dev)
        model.fit(data, epochs=args.epochs, device=dev)

    # 7. Inference & Trajectory Evaluation
    x0_eval = torch.randn(args.n_samples, 2, device=dev)
    apply_inf_repulsion = args.obstacle_mode in ["inference", "both"]

    steered_streamer = RiemannianRepulsionStreamer(
        flow_model=model,
        potential=potential if apply_inf_repulsion else None,
        metric=base_metric,
        alpha=args.alpha if apply_inf_repulsion else 0.0,
        gyro_curl=args.gyro_curl if apply_inf_repulsion else 0.0,
    )

    if args.model == "rmeanfm" and not apply_inf_repulsion:
        x1_gen, traj = model.sample(x0_eval, steps=5)
    else:
        x1_gen, traj = integrate_ode(steered_streamer, x0_eval, steps=args.ode_steps, solver="rk4")

    # Compute Intrinsic Core Penetration, Endpoint Violation Rates & MMD Comparison
    with torch.no_grad():
        T_steps, B_eval, D_dim = traj.shape
        dists_intrinsic = potential.compute_distance(traj.view(T_steps * B_eval, D_dim)).view(T_steps, B_eval)
    path_viol = (dists_intrinsic.min(dim=0)[0] < r_min).float().mean().item() * 100.0
    end_viol = (dists_intrinsic[-1] < r_min).float().mean().item() * 100.0
    mmd_val = compute_mmd_rbf(x1_gen, safe_gt_data)
    elapsed = time.perf_counter() - t_start

    print("-" * 75)
    print(f"[Evaluation] Path Violations (< r_min): {path_viol:.2f}% | Endpoint Violations: {end_viol:.2f}%")
    print(f"[MMD Metric] Model Safe MMD: {mmd_val:.5f} | Oracle Half-Split MMD: {oracle_half_mmd:.5f} (Ratio: {mmd_val / oracle_half_mmd:.2f}x)")
    print(f"[Runtime]    Total Elapsed:  {elapsed:.1f}s")
    print("-" * 75)

    # 8. Register Run into Main CSV File
    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.exists(args.csv_path)

    next_idx = 1
    if file_exists:
        with open(args.csv_path, mode="r", encoding="utf-8") as f_in:
            rows = list(csv.reader(f_in))
            if len(rows) > 1:
                try:
                    next_idx = max(int(r[0]) for r in rows[1:] if r and r[0].isdigit()) + 1
                except ValueError:
                    next_idx = len(rows)

    group_name = "euclidean_baseline" if is_euc_base else "riemannian"
    with open(args.csv_path, mode="a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if not file_exists:
            writer.writerow([
                "index", "group", "filename", "model", "metric", "premetric",
                "ot", "ot_method", "obstacle_mode", "path_viol_pct", "end_viol_pct",
                "safe_mmd", "oracle_half_mmd", "runtime_sec", "status"
            ])
        writer.writerow([
            next_idx, group_name, os.path.basename(out_path), args.model, args.metric, args.premetric,
            args.ot, args.ot_method, args.obstacle_mode,
            f"{path_viol:.2f}", f"{end_viol:.2f}", f"{mmd_val:.5f}", f"{oracle_half_mmd:.5f}", f"{elapsed:.1f}", "SUCCESS"
        ])
    print(f"Registered run #{next_idx} into {args.csv_path}")

    # 9. Plot 3-Panel Diagnostics
    title_str = (
        f"{args.model.upper()} ({args.metric.upper()}, {args.premetric}, {args.obstacle_mode}) | "
        f"Path Viol: {path_viol:.1f}%, End Viol: {end_viol:.1f}%, MMD: {mmd_val:.3f} (Oracle: {oracle_half_mmd:.3f})"
    )
    plot_full_diagnostics(
        data=data,
        trajectories=traj,
        forbidden_center=forbid_center,
        r_min=r_min,
        r_max=r_max,
        steered_field=steered_streamer,
        metric=base_metric,
        interpolant_net=interpolant_net,
        potential=potential,
        show_potential_heatmap=(args.obstacle_mode != "none"),
        title=title_str,
        output_path=out_path,
    )
    if args.plot_3d:
        out_3d_path = out_path.replace(".png", "_3D.png")
        plot_3d_manifold_diagnostics(
            data=data,
            trajectories=traj,
            forbidden_center=forbid_center,
            r_min=r_min,
            r_max=r_max,
            metric=base_metric,
            interpolant_net=interpolant_net,
            potential=potential,
            steered_field=steered_streamer,
            title=f"{args.model.upper()} ({args.metric.upper()}, {args.obstacle_mode})",
            output_path=out_3d_path,
        )


if __name__ == "__main__":
    main()