import os
import csv
import time
import copy
import argparse
import traceback
from typing import Dict, List, Tuple

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
from steering.repulsion import (
    CompactForbiddenPotential,
    RiemannianRepulsionStreamer,
    integrate_ode,
)
from utils.visualization import plot_full_diagnostics


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.25, 0.5, 1.0)) -> float:
    """Computes multi-scale Gaussian RBF Maximum Mean Discrepancy (MMD) to safe ground-truth data."""
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


def build_experiment_grid(include_sinkhorn: bool = False) -> List[Dict[str, str]]:
    """
    Builds a focused, scientifically meaningful benchmark grid:
      1. Euclidean Control Baseline (1 per model x obstacle_mode):
         metric='none', premetric='euclidean', ot='euclidean' (exact).
         Demonstrates why Euclidean geometry fails on curved manifolds & obstacles.
      2. True Riemannian Suite:
         - Metrics: strictly Riemannian ('cdc', 'land', 'rbf')
         - Premetrics:
             * For RFM: 'diffusion', 'biharmonic'
             * For MFM & RMeanFM: 'geodesic' (amortized MFM arc-length)
         - OT: 'none' vs 'premetric' (using the true Riemannian distance cost!)
    """
    models = ["rfm", "mfm", "rmeanfm"]
    riemannian_metrics = ["cdc", "land", "rbf"]
    obstacle_modes = ["none", "baked", "inference", "both"]

    ot_riemannian_configs = [("none", "none"), ("premetric", "exact")]
    if include_sinkhorn:
        ot_riemannian_configs.append(("premetric", "sinkhorn"))

    grid = []

    for model_name in models:
        # --- A. 1x Euclidean Control Baseline per Model ---
        for obs_mode in obstacle_modes:
            # Note: RFM with metric='none' has no metric tensor to warp in 'baked' mode,
            # so for RFM Euclidean baseline we test 'none' (unsteered) and 'inference' (Euclidean repulsion).
            if model_name == "rfm" and obs_mode in ["baked", "both"]:
                continue
            grid.append(
                {
                    "group": "euclidean_baseline",
                    "model": model_name,
                    "metric": "none",
                    "premetric": "euclidean",
                    "ot": "euclidean",
                    "ot_method": "exact",
                    "obstacle": obs_mode,
                }
            )

        # --- B. True Riemannian Configurations ---
        for metric_name in riemannian_metrics:
            valid_premetrics = ["diffusion", "biharmonic"] if model_name == "rfm" else ["geodesic"]

            for premetric_name in valid_premetrics:
                for ot_mode, ot_method in ot_riemannian_configs:
                    for obs_mode in obstacle_modes:
                        grid.append(
                            {
                                "group": "riemannian",
                                "model": model_name,
                                "metric": metric_name,
                                "premetric": premetric_name,
                                "ot": ot_mode,
                                "ot_method": ot_method,
                                "obstacle": obs_mode,
                            }
                        )

    return grid


def make_filename(cfg: Dict[str, str]) -> str:
    """Produces clean, descriptive filename."""
    ot_str = "ot_none" if cfg["ot"] == "none" else f"ot_{cfg['ot']}_{cfg['ot_method']}"
    prefix = "BASELINE_EUC_" if cfg["group"] == "euclidean_baseline" else ""
    return f"{prefix}{cfg['model']}_{cfg['metric']}_{cfg['premetric']}_{ot_str}_{cfg['obstacle']}.png"


def get_top_moon_peak(raw_x: torch.Tensor, raw_y: torch.Tensor, data_std_x: torch.Tensor) -> torch.Tensor:
    """
    Finds the exact standardized coordinates of the crest/peak of the upper half-moon (label == 0).
    In raw make_moons, the top moon peak is at (0.0, 1.0), which maps to approx (-0.575, +1.48)
    in standardized coordinates.
    """
    top_moon = data_std_x[raw_y == 0]
    # Select points near the horizontal apex of the upper arch and compute their centroid
    top_apex_mask = top_moon[:, 1] >= torch.quantile(top_moon[:, 1], 0.92)
    peak_coord = top_moon[top_apex_mask].mean(dim=0)
    return peak_coord


def main():
    parser = argparse.ArgumentParser(description="Focused Benchmark Sweep: Euclidean Baseline vs. Riemannian Flows")
    parser.add_argument("--results-dir", type=str, default="./results_sweep", help="Directory to save all plots and CSV")
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs for flow models")
    parser.add_argument("--epochs-cdc", type=int, default=800, help="Training epochs for CDC metric (cached once)")
    parser.add_argument("--ode-steps", type=int, default=100, help="RK4 integration steps at test time")
    parser.add_argument("--n-samples", type=int, default=600, help="Evaluation samples per run")
    parser.add_argument("--alpha", type=float, default=1.2, help="Inference repulsion strength")
    parser.add_argument("--gyro-curl", type=float, default=0.35, help="Gyroscopic circulation weight")
    parser.add_argument("--include-sinkhorn", action="store_true", help="Also test Sinkhorn OT in addition to Exact OT")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if output PNG already exists")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "summary_results.csv")
    dev = torch.device(args.device)

    # 1. Fixed Seed & Standardized 2-Moons Dataset
    torch.manual_seed(42)
    raw_x, raw_y = make_moons(n_samples=4000, noise=0.07, random_state=42)
    raw_x_t = torch.tensor(raw_x, dtype=torch.float32, device=dev)
    raw_y_t = torch.tensor(raw_y, dtype=torch.long, device=dev)
    data = (raw_x_t - raw_x_t.mean(dim=0)) / raw_x_t.std(dim=0)

    # Place Forbidden Area directly at the peak/apex of the top half-moon!
    forbid_center = get_top_moon_peak(raw_x_t, raw_y_t, data).to(dev)
    r_min, r_max = 0.32, 0.70
    print(f"Forbidden Zone placed at Top Moon Peak: center = ({forbid_center[0]:.3f}, {forbid_center[1]:.3f}), r_min={r_min}, r_max={r_max}")

    # Reference safe ground-truth data (points outside r_min) for MMD calculation
    dists_gt = torch.linalg.norm(data - forbid_center.unsqueeze(0), dim=-1)
    safe_gt_data = data[dists_gt >= r_min]

    # 2. Precompute & Cache Base Metrics Once!
    print("\n" + "=" * 75)
    print("[Phase 1/3] Pre-computing & Caching Base Riemannian Metrics (CDC, LAND, RBF)...")
    print("=" * 75)
    t0_init = time.perf_counter()

    cdc_metric = CDCMetric(dim=2, rank=2, hidden_dim=128, lambda_reg=1e-2).to(dev)
    cdc_cache_path = os.path.join(args.results_dir, "cached_cdc_metric.pt")
    if os.path.exists(cdc_cache_path) and not args.overwrite:
        print(f"Loading cached CDC Metric from {cdc_cache_path}...")
        cdc_metric.load_state_dict(torch.load(cdc_cache_path, map_location=dev))
        cdc_metric.eval()
    else:
        print("Training CDC Metric Matching Network...")
        cdc_metric.fit(data, epochs=args.epochs_cdc, batch_size=256, lr=1e-3, device=dev)
        torch.save(cdc_metric.state_dict(), cdc_cache_path)

    metrics_cache = {
        "cdc": cdc_metric,
        "land": LANDMetric(data=data, max_Reference_points=600, sigma=0.25, lambda_reg=1e-2).to(dev),
        "rbf": RBFMetric(data=data, num_centers=80, bandwidth=None, lambda_reg=1e-2).to(dev),
        "none": None,
    }

    # 3. Precompute & Cache Static Premetrics Once!
    print("[Phase 2/3] Pre-computing Spectral Laplacian Premetrics (Diffusion & Biharmonic)...")
    static_premetrics = {
        "euclidean": EuclideanPremetric().to(dev),
        "diffusion": SpectralLaplacianPremetric(data=data, mode="diffusion", num_landmarks=512, num_eigen=32, sigma=0.35).to(dev),
        "biharmonic": SpectralLaplacianPremetric(data=data, mode="biharmonic", num_landmarks=512, num_eigen=32, sigma=0.35).to(dev),
        "none": None,
    }
    print(f"Initialization complete in {time.perf_counter() - t0_init:.1f}s.")

    # Cache for MFM Stage 1 Geodesic Interpolants: key = (metric_name, is_baked)
    interpolant_cache: Dict[Tuple[str, bool], torch.nn.Module] = {}

    # 4. Build Focused Grid & Run Benchmark
    grid = build_experiment_grid(include_sinkhorn=args.include_sinkhorn)
    print("\n" + "=" * 75)
    print(f"[Phase 3/3] Starting Focused Benchmark Sweep: {len(grid)} Meaningful Configurations")
    print("=" * 75)

    write_header = not os.path.exists(csv_path) or args.overwrite
    csv_file = open(csv_path, mode="w" if write_header else "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "index", "group", "filename", "model", "metric", "premetric", "ot", "ot_method", "obstacle_mode",
            "path_viol_pct", "end_viol_pct", "safe_mmd", "runtime_sec", "status"
        ])
        csv_file.flush()

    for idx, cfg in enumerate(grid, start=1):
        fname = make_filename(cfg)
        out_path = os.path.join(args.results_dir, fname)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{idx}/{len(grid)}] SKIPPING (exists): {fname}")
            continue

        print(f"\n[{idx}/{len(grid)}] RUNNING ({cfg['group']}): {fname}")
        t_start = time.perf_counter()

        try:
            torch.manual_seed(42)
            base_metric = metrics_cache[cfg["metric"]]
            is_baked = cfg["obstacle"] in ["baked", "both"]
            use_inf_repulsion = cfg["obstacle"] in ["inference", "both"]

            # Select initial premetric
            premetric = static_premetrics.get(cfg["premetric"], static_premetrics["euclidean"])

            # Create Forbidden Potential at the Top Moon Peak
            potential = CompactForbiddenPotential(
                center=forbid_center,
                r_min=r_min,
                r_max=r_max,
                scale=18.0,
                premetric=premetric,
            ).to(dev)

            # Configure active training metric
            active_metric = base_metric
            if is_baked and base_metric is not None and cfg["model"] == "rfm":
                active_metric = ConformalObstacleMetric(base_metric=base_metric, potential_fn=potential, beta=1.0).to(dev)

            # Configure initial OT Coupler
            if cfg["ot"] == "none":
                coupler = IndependentCoupler()
            elif cfg["ot"] == "euclidean":
                coupler = MinibatchOTCoupler(premetric=static_premetrics["euclidean"], method=cfg["ot_method"])
            else:
                coupler = MinibatchOTCoupler(premetric=premetric, method=cfg["ot_method"])

            interpolant_net = None

            # --- Train Model ---
            if cfg["model"] == "rfm":
                model = RiemannianFlowMatching(
                    dim=2,
                    metric=active_metric,
                    premetric=premetric,
                    coupler=coupler,
                    simulate_premetric_ode=False,
                ).to(dev)
                model.fit(data, epochs=args.epochs, device=dev)

            elif cfg["model"] in ["mfm", "rmeanfm"]:
                cache_key = (cfg["metric"], is_baked)
                if cache_key in interpolant_cache:
                    print(f"  -> Using cached MFM Stage 1 Geodesic Interpolant for {cache_key}")
                    interpolant_net = copy.deepcopy(interpolant_cache[cache_key]).to(dev)
                else:
                    baked_pot = potential if is_baked else None
                    mfm_init = MetricFlowMatching(
                        dim=2, metric=active_metric, potential=baked_pot, coupler=coupler
                    ).to(dev)
                    mfm_init.fit_interpolant(data, epochs=int(args.epochs * 0.75), device=dev)
                    interpolant_net = mfm_init.interpolant_net
                    interpolant_cache[cache_key] = copy.deepcopy(interpolant_net).cpu()

                # Upgrade to MFMGeodesicPremetric for Riemannian runs
                if base_metric is not None and cfg["premetric"] == "geodesic":
                    geo_premetric = MFMGeodesicPremetric(
                        mfm_interpolant=interpolant_net, metric=base_metric, num_quad_steps=12
                    ).to(dev)
                    potential.premetric = geo_premetric
                    if cfg["ot"] == "premetric":
                        coupler = MinibatchOTCoupler(premetric=geo_premetric, method=cfg["ot_method"])

                if cfg["model"] == "mfm":
                    model = MetricFlowMatching(
                        dim=2, metric=active_metric, potential=potential if is_baked else None, coupler=coupler
                    ).to(dev)
                    model.interpolant_net = interpolant_net
                    model.fit_flow(data, epochs=args.epochs, device=dev)
                else:
                    model = RiemannianMeanFlow(
                        dim=2, interpolant_net=interpolant_net, metric=active_metric, coupler=coupler
                    ).to(dev)
                    model.fit(data, epochs=args.epochs, device=dev)

            # --- Evaluate & Integrate ---
            x0_eval = torch.randn(args.n_samples, 2, device=dev)
            steered_streamer = RiemannianRepulsionStreamer(
                flow_model=model,
                potential=potential if use_inf_repulsion else None,
                metric=base_metric,
                alpha=args.alpha if use_inf_repulsion else 0.0,
                gyro_curl=args.gyro_curl if use_inf_repulsion else 0.0,
            )

            if cfg["model"] == "rmeanfm" and not use_inf_repulsion:
                x1_gen, traj = model.sample(x0_eval, steps=5)
            else:
                x1_gen, traj = integrate_ode(steered_streamer, x0_eval, steps=args.ode_steps, solver="rk4")

            dists = torch.linalg.norm(traj - forbid_center.view(1, 1, -1), dim=-1)
            path_viol = (dists.min(dim=0)[0] < r_min).float().mean().item() * 100.0
            end_viol = (dists[-1] < r_min).float().mean().item() * 100.0
            mmd_val = compute_mmd_rbf(x1_gen, safe_gt_data)
            elapsed = time.perf_counter() - t_start

            title_str = (
                f"{cfg['model'].upper()} ({cfg['metric'].upper()}, {cfg['premetric']}, {cfg['obstacle']}) | "
                f"Path Viol: {path_viol:.1f}%, End Viol: {end_viol:.1f}%"
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
                potential=potential if cfg["obstacle"] != "none" else None,
                title=title_str,
                output_path=out_path,
            )

            csv_writer.writerow([
                idx, cfg["group"], fname, cfg["model"], cfg["metric"], cfg["premetric"],
                cfg["ot"], cfg["ot_method"], cfg["obstacle"],
                f"{path_viol:.2f}", f"{end_viol:.2f}", f"{mmd_val:.5f}", f"{elapsed:.1f}", "SUCCESS"
            ])
            csv_file.flush()
            print(f"  -> Done in {elapsed:.1f}s | Path Viol: {path_viol:.1f}% | End Viol: {end_viol:.1f}% | MMD: {mmd_val:.4f}")

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            print(f"  -> ERROR in {fname}: {e}")
            traceback.print_exc()
            csv_writer.writerow([
                idx, cfg["group"], fname, cfg["model"], cfg["metric"], cfg["premetric"],
                cfg["ot"], cfg["ot_method"], cfg["obstacle"],
                "NaN", "NaN", "NaN", f"{elapsed:.1f}", f"ERROR: {str(e)}"
            ])
            csv_file.flush()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_file.close()
    print("\nFocused sweep completed! Summary saved to:", csv_path)


if __name__ == "__main__":
    main()