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
from utils.visualization import plot_full_diagnostics


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
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=1.2, help="Inference repulsion strength")
    parser.add_argument("--gyro-curl", type=float, default=0.35, help="Gyroscopic circulation to prevent stuck flows")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="riemannian_demo.png")
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    print("=" * 72)
    print(f"Model: {args.model.upper()} | Metric: {args.metric.upper()} | Premetric: {args.premetric.upper()}")
    print(f"OT: {args.ot.upper()} ({args.ot_method}) | Obstacle Mode: {args.obstacle_mode.upper()} | Device: {dev}")
    print("=" * 72)

    # 1. Generate Standardized 2-Moons Dataset
    raw_x, _ = make_moons(n_samples=4000, noise=0.07, random_state=42)
    data = torch.tensor(raw_x, dtype=torch.float32, device=dev)
    data = (data - data.mean(dim=0)) / data.std(dim=0)

    # 2. Initialize & Fit Base Riemannian Metric G(x)
    base_metric = None
    if args.metric == "cdc":
        base_metric = CDCMetric(dim=2, rank=2, hidden_dim=128, lambda_reg=1e-2).to(dev)
        base_metric.fit(data, epochs=args.epochs, batch_size=256, lr=1e-3, device=dev)
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

    # 4. Forbidden Region & Compact Potential V_t(x)
    forbid_center = torch.tensor([-0.50, 0.75], device=dev)
    r_min, r_max = 0.28, 0.65
    potential = CompactForbiddenPotential(
        center=forbid_center,
        r_min=r_min,
        r_max=r_max,
        scale=18.0,
        premetric=premetric,
    ).to(dev)

    # Conformal metric wrapper for baking obstacle into RFM metric G_tilde(x) = (1 + beta V(x)) G(x)
    active_metric = base_metric
    if args.obstacle_mode in ["baked", "both"] and base_metric is not None and args.model == "rfm":
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
        baked_pot = potential if args.obstacle_mode in ["baked", "both"] else None
        model = MetricFlowMatching(
            dim=2,
            metric=active_metric,
            potential=baked_pot,
            coupler=coupler,
        ).to(dev)
        # Stage 1: Minimize Kinetic Energy + Forbidden Potential V_t(gamma_eta(t))
        model.fit_interpolant(data, epochs=int(args.epochs * 0.75), device=dev)
        interpolant_net = model.interpolant_net

        # Wrap the trained MFM Geodesic Interpolant as a fast amortized Geodesic Premetric!
        if base_metric is not None:
            geo_premetric = MFMGeodesicPremetric(
                mfm_interpolant=interpolant_net, metric=base_metric, num_quad_steps=12
            ).to(dev)
            if args.premetric == "geodesic":
                potential.premetric = geo_premetric
                if args.ot == "premetric":
                    model.coupler = MinibatchOTCoupler(premetric=geo_premetric, method=args.ot_method)

        # Stage 2: Train Velocity Field v_theta (drops metric at inference time!)
        model.fit_flow(data, epochs=args.epochs, device=dev)

    elif args.model == "rmeanfm":
        baked_pot = potential if args.obstacle_mode in ["baked", "both"] else None
        mfm_stage1 = MetricFlowMatching(
            dim=2, metric=active_metric, potential=baked_pot, coupler=coupler
        ).to(dev)
        mfm_stage1.fit_interpolant(data, epochs=int(args.epochs * 0.65), device=dev)
        interpolant_net = mfm_stage1.interpolant_net

        if base_metric is not None and args.premetric == "geodesic":
            geo_premetric = MFMGeodesicPremetric(
                mfm_interpolant=interpolant_net, metric=base_metric, num_quad_steps=12
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

    # Compute Core Penetration & Endpoint Violation Rates
    dists = torch.linalg.norm(traj - forbid_center.view(1, 1, -1), dim=-1)
    path_viol = (dists.min(dim=0)[0] < r_min).float().mean().item() * 100.0
    end_viol = (dists[-1] < r_min).float().mean().item() * 100.0
    print(f"\n[Evaluation] Path Violations (< r_min): {path_viol:.2f}% | Endpoint Violations: {end_viol:.2f}%")

    # 8. Plot 3-Panel Diagnostics
    plot_full_diagnostics(
        data=data,
        trajectories=traj,
        forbidden_center=forbid_center,
        r_min=r_min,
        r_max=r_max,
        steered_field=steered_streamer,
        metric=base_metric,
        interpolant_net=interpolant_net,
        potential=potential if args.obstacle_mode != "none" else None,
        title=f"{args.model.upper()} ({args.metric.upper()}) | Path Viol: {path_viol:.1f}%, End Viol: {end_viol:.1f}%",
        output_path=args.output,
    )


if __name__ == "__main__":
    main()