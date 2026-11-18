import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from geometry.metrics import (
    CDCMetric,
    LANDMetric,
    RBFMetric,
    VAEPullbackMetric,
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
    KMeansForbiddenPotential,
    ClassifierForbiddenPotential,
    RiemannianRepulsionStreamer,
    integrate_ode,
)


# =====================================================================
# 1. Dimensionality Reduction & Generative Decoders (PCA & VAE)
# =====================================================================
class PCALatentSpace(nn.Module):
    """Exact SVD-based PCA mapping 784-dim MNIST images to standardized R^d latents."""

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(1, 784))
        self.register_buffer("components", torch.zeros(784, latent_dim))
        self.register_buffer("latent_std", torch.ones(1, latent_dim))

    @torch.no_grad()
    def fit(self, x_flat: torch.Tensor):
        mu = x_flat.mean(dim=0, keepdim=True)
        x_centered = x_flat - mu
        _, _, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        comps = Vh[: self.latent_dim].T  # [784, d]
        z_raw = x_centered @ comps
        z_std = z_raw.std(dim=0, keepdim=True).clamp(min=1e-6)

        self.mean.copy_(mu)
        self.components.copy_(comps)
        self.latent_std.copy_(z_std)

    @torch.no_grad()
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        return ((x_flat - self.mean) @ self.components) / self.latent_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_rec = (z * self.latent_std) @ self.components.T + self.mean
        return torch.clamp(x_rec, 0.0, 1.0)


class MNISTVAE(nn.Module):
    """Smooth MLP Variational Autoencoder with SiLU activations (C^2 smooth for Pullback Metric)."""

    def __init__(self, latent_dim: int = 2, hidden_dim: int = 400):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid(),
        )
        self.register_buffer("z_mean", torch.zeros(1, latent_dim))
        self.register_buffer("z_std", torch.ones(1, latent_dim))

    def encode_raw(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_raw(x)
        return (mu - self.z_mean) / self.z_std

    def decode(self, z_norm: torch.Tensor) -> torch.Tensor:
        z_raw = z_norm * self.z_std + self.z_mean
        return self.decoder(z_raw)

    def fit(
        self,
        x_data: torch.Tensor,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta_kl: float = 0.5,
        device: str = "cpu",
    ):
        self.to(device)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data), batch_size=batch_size, shuffle=True
        )
        self.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(device)
                mu, logvar = self.encode_raw(xb)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_rec = self.decoder(z)

                recon_loss = (
                    nn.functional.mse_loss(x_rec, xb, reduction="sum") / xb.shape[0]
                )
                kl_loss = (
                    -0.5
                    * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                )
                loss = recon_loss + beta_kl * kl_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        with torch.no_grad():
            all_mu, _ = self.encode_raw(x_data.to(device))
            self.z_mean.copy_(all_mu.mean(dim=0, keepdim=True))
            self.z_std.copy_(all_mu.std(dim=0, keepdim=True).clamp(min=1e-6))


class PixelOracleClassifier(nn.Module):
    """Independent 784-dim Pixel Classifier to evaluate whether decoded images belong to the forbidden digit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        epochs: int = 15,
        device: str = "cpu",
    ):
        self.to(device)
        opt = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data, y_data), batch_size=256, shuffle=True
        )
        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(self.net(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.eval()


def compute_mmd_rbf(
    x: torch.Tensor, y: torch.Tensor, sigmas=(0.2, 0.5, 1.0, 2.0)
) -> float:
    with torch.no_grad():
        dxx = torch.cdist(x, x, p=2.0) ** 2
        dyy = torch.cdist(y, y, p=2.0) ** 2
        dxy = torch.cdist(x, y, p=2.0) ** 2
        kxx = sum(torch.exp(-dxx / (2 * s * s)) for s in sigmas)
        kyy = sum(torch.exp(-dyy / (2 * s * s)) for s in sigmas)
        kxy = sum(torch.exp(-dxy / (2 * s * s)) for s in sigmas)
        return max((kxx.mean() + kyy.mean() - 2 * kxy.mean()).item(), 0.0)


# =====================================================================
# 2. Visualization Suite for Latent Manifold + Decoded MNIST Digits
# =====================================================================
@torch.no_grad()
def plot_mnist_avoidance_results(
    latents: torch.Tensor,
    labels: torch.Tensor,
    traj: torch.Tensor,
    decoded_imgs: torch.Tensor,
    pred_classes: torch.Tensor,
    pred_probs: torch.Tensor,
    potential: nn.Module,
    forbidden_digit: int,
    title: str,
    output_path: str,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    device = latents.device

    D = latents.shape[-1]
    if D > 2:
        mu_z = latents.mean(dim=0, keepdim=True)
        _, _, Vh_z = torch.linalg.svd(latents - mu_z, full_matrices=False)
        P2 = Vh_z[:2].T  # [D, 2]
        z_2d = ((latents - mu_z) @ P2).cpu().numpy()
        T_s, B_s, _ = traj.shape
        traj_2d = (
            (((traj.reshape(T_s * B_s, D) - mu_z) @ P2).reshape(T_s, B_s, 2))
            .cpu()
            .numpy()
        )
    else:
        P2 = None
        mu_z = torch.zeros(1, 2, device=device)
        z_2d = latents.cpu().numpy()
        traj_2d = traj.cpu().numpy()

    y_np = labels.cpu().numpy()
    fig = plt.figure(figsize=(22, 7.0))

    gx, gy = np.meshgrid(np.linspace(-2.8, 2.8, 90), np.linspace(-2.8, 2.8, 90))
    grid_2d = torch.tensor(
        np.stack([gx.ravel(), gy.ravel()], axis=-1), dtype=torch.float32, device=device
    )
    if P2 is not None:
        c_full = potential.center.reshape(1, D)
        c_2d = (c_full - mu_z) @ P2
        grid_full = c_full + (grid_2d - c_2d) @ P2.T
    else:
        grid_full = grid_2d

    b_grid = potential.normalized_intensity(grid_full).cpu().numpy().reshape(90, 90)

    # --- Panel 1: 10-Class MNIST Latent Space & Barrier Contours ---
    ax0 = fig.add_subplot(1, 3, 1)
    cmap_digits = plt.get_cmap("tab10")
    for digit in range(10):
        mask = y_np == digit
        alpha_pt = 0.85 if digit == forbidden_digit else 0.35
        marker_sz = 10 if digit == forbidden_digit else 4
        lbl = (
            f"Digit {digit} (FORBIDDEN)"
            if digit == forbidden_digit
            else f"Digit {digit}"
        )
        ax0.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            s=marker_sz,
            color=cmap_digits(digit),
            alpha=alpha_pt,
            label=lbl,
        )

    ax0.contourf(
        gx, gy, b_grid, levels=np.linspace(0.05, 1.0, 12), cmap="Reds", alpha=0.38
    )
    ax0.contour(
        gx,
        gy,
        b_grid,
        levels=[0.85],
        colors=["red"],
        linewidths=2.2,
        linestyles="solid",
    )
    ax0.contour(
        gx,
        gy,
        b_grid,
        levels=[0.05],
        colors=["darkorange"],
        linewidths=1.5,
        linestyles="dotted",
    )
    ax0.set_title(
        f"1. Latent Space & Barrier Shape (Forbidden Digit = {forbidden_digit})",
        fontsize=12,
        fontweight="bold",
    )
    ax0.set_xlim(-2.8, 2.8)
    ax0.set_ylim(-2.8, 2.8)
    ax0.legend(loc="upper right", fontsize=7.5, ncol=2)

    # --- Panel 2: Steered Latent Trajectories ---
    ax1 = fig.add_subplot(1, 3, 2)
    safe_mask = y_np != forbidden_digit
    forb_mask = y_np == forbidden_digit
    ax1.scatter(
        z_2d[safe_mask, 0],
        z_2d[safe_mask, 1],
        s=3,
        c="lightgray",
        alpha=0.45,
        label="Safe Digits Support",
    )
    ax1.scatter(
        z_2d[forb_mask, 0],
        z_2d[forb_mask, 1],
        s=6,
        c="salmon",
        alpha=0.45,
        label=f"Forbidden Digit {forbidden_digit}",
    )
    ax1.contour(
        gx,
        gy,
        b_grid,
        levels=[0.85],
        colors=["red"],
        linewidths=2.2,
        linestyles="solid",
    )
    ax1.contour(
        gx,
        gy,
        b_grid,
        levels=[0.05],
        colors=["darkorange"],
        linewidths=1.5,
        linestyles="dotted",
    )

    n_plot = min(75, traj_2d.shape[1])
    for b in range(n_plot):
        lx, ly = traj_2d[:, b, 0], traj_2d[:, b, 1]
        ax1.plot(lx, ly, color="forestgreen", alpha=0.45, lw=1.2)
        ax1.scatter(lx[0], ly[0], marker="x", color="black", s=14, alpha=0.45)
        ax1.scatter(lx[-1], ly[-1], marker="o", color="darkgreen", s=18, alpha=0.9)

    ax1.set_title(
        f"2. Latent Flow Trajectories\n{title}", fontsize=11.5, fontweight="bold"
    )
    ax1.set_xlim(-2.8, 2.8)
    ax1.set_ylim(-2.8, 2.8)
    ax1.legend(loc="upper left", fontsize=8.5)

    # --- Panel 3: 8x8 Grid of Decoded MNIST Images ---
    ax2 = fig.add_subplot(1, 3, 3)
    grid_k = 8
    canvas = np.ones((grid_k * 28, grid_k * 28), dtype=np.float32)
    imgs_np = decoded_imgs[: grid_k * grid_k].cpu().numpy().reshape(-1, 28, 28)
    preds_np = pred_classes[: grid_k * grid_k].cpu().numpy()

    for idx in range(min(grid_k * grid_k, len(imgs_np))):
        r, c = divmod(idx, grid_k)
        canvas[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = 1.0 - imgs_np[idx]

    ax2.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    for idx in range(min(grid_k * grid_k, len(imgs_np))):
        r, c = divmod(idx, grid_k)
        is_forb = preds_np[idx] == forbidden_digit
        txt_col = "red" if is_forb else "navy"
        ax2.text(
            c * 28 + 2,
            r * 28 + 7,
            str(preds_np[idx]),
            color=txt_col,
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="mistyrose" if is_forb else "white",
                alpha=0.75,
                edgecolor="none",
            ),
        )
    ax2.set_xticks([])
    ax2.set_yticks([])
    forb_rate = (pred_classes == forbidden_digit).float().mean().item() * 100.0
    ax2.set_title(
        f"3. Decoded MNIST Samples (Oracle Classifier Labels)\nGenerated Digit '{forbidden_digit}' Rate: {forb_rate:.2f}% (Baseline ~10.0%)",
        fontsize=11.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved MNIST diagnostic plot to {output_path}")


# =====================================================================
# 3. Main CLI Runner with Full Model & Barrier Checkpoint Caching
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MNIST Latent Class Avoidance: PCA vs. VAE Pullback & 3 Barrier Shapes"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="vae",
        choices=["pca", "vae"],
        help="Latent space type",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="Dimension D of the latent space (e.g. 2, 4, 8, 16)",
    )
    parser.add_argument(
        "--forbidden-digit",
        type=int,
        default=3,
        help="MNIST digit class (0-9) to forbid",
    )
    parser.add_argument(
        "--barrier",
        type=str,
        default="kmeans",
        choices=["centroid", "kmeans", "classifier"],
        help="Shape of the forbidden barrier: single centroid hyperball, k-means union, or MLP classifier",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=6,
        help="Number of sub-centroids for --barrier kmeans",
    )
    parser.add_argument(
        "--model", type=str, default="mfm", choices=["rfm", "mfm", "rmeanfm"]
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pullback",
        choices=["pullback", "cdc", "land", "rbf", "none"],
    )
    parser.add_argument(
        "--premetric",
        type=str,
        default="geodesic",
        choices=["euclidean", "diffusion", "biharmonic", "geodesic"],
    )
    parser.add_argument(
        "--ot",
        type=str,
        default="premetric",
        choices=["none", "euclidean", "premetric"],
    )
    parser.add_argument(
        "--ot-method", type=str, default="exact", choices=["exact", "sinkhorn"]
    )
    parser.add_argument(
        "--obstacle-mode",
        type=str,
        default="inference",
        choices=["none", "baked", "inference", "both"],
        help="Use 'inference' (or 'none') to share a single trained unbaked model across all barriers & digits!",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--n-train",
        type=int,
        default=6000,
        help="Number of MNIST training images to use",
    )
    parser.add_argument(
        "--n-samples", type=int, default=600, help="Number of test samples to generate"
    )
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--gyro-curl", type=float, default=0.65)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if checkpoint exists on disk",
    )
    parser.add_argument("--results-dir", type=str, default="./results_mnist")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    t_start = time.perf_counter()

    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    use_cache = not args.force_retrain

    # If embedding is PCA, pullback is Euclidean I_d, so default 'pullback' to 'cdc' on PCA latents
    eff_metric = (
        "cdc"
        if (args.embedding == "pca" and args.metric == "pullback")
        else args.metric
    )
    ot_str = "ot_none" if args.ot == "none" else f"ot_{args.ot}_{args.ot_method}"

    fname = (
        f"mnist_{args.embedding}_d{args.latent_dim}_forbid{args.forbidden_digit}_"
        f"{args.barrier}_{args.model}_{eff_metric}_{ot_str}_{args.obstacle_mode}.png"
    )
    out_path = os.path.join(args.results_dir, fname)

    # Smart Checkpoint Naming:
    # - When obstacle_mode in ['none', 'inference'], the trained model is UNBAKED and completely
    #   independent of `--barrier` and `--forbidden-digit`! Thus 1 trained model serves ALL barriers & digits!
    # - When obstacle_mode in ['baked', 'both'], the potential V(z) is baked into training, so the checkpoint
    #   is tagged with the specific barrier and forbidden digit.
    is_baked = args.obstacle_mode in ["baked", "both"]
    if is_baked:
        bake_tag = f"baked_{args.barrier}_forbid{args.forbidden_digit}"
    else:
        bake_tag = "unbaked"

    interp_ckpt = os.path.join(
        ckpt_dir,
        f"interp_{args.embedding}_d{args.latent_dim}_{eff_metric}_{bake_tag}_ep{args.epochs}.pt",
    )
    flow_ckpt = os.path.join(
        ckpt_dir,
        f"flow_{args.model}_{args.embedding}_d{args.latent_dim}_{eff_metric}_{args.premetric}_{ot_str}_{bake_tag}_ep{args.epochs}.pt",
    )

    print("=" * 85)
    print(
        f"MNIST Class Avoidance | Embedding: {args.embedding.upper()} (D={args.latent_dim}) | Forbid Digit: {args.forbidden_digit}"
    )
    print(
        f"Barrier Shape: {args.barrier.upper()} | Model: {args.model.upper()} | Metric: {eff_metric.upper()} | Mode: {args.obstacle_mode.upper()}"
    )
    print(f"Flow Checkpoint: {os.path.basename(flow_ckpt)} (Auto-Cache: {use_cache})")
    print("=" * 85)

    # 1. Load MNIST Dataset
    torch.manual_seed(42)
    mnist_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    idx_sub = torch.randperm(len(mnist_ds))[: args.n_train]
    x_pixels = mnist_ds.data[idx_sub].float().reshape(-1, 784) / 255.0
    y_labels = mnist_ds.targets[idx_sub].long()
    x_pixels, y_labels = x_pixels.to(dev), y_labels.to(dev)

    # 2. Train or Load Pixel Oracle Classifier
    oracle_clf = PixelOracleClassifier().to(dev)
    oracle_ckpt = os.path.join(ckpt_dir, "pixel_oracle_clf.pt")
    if use_cache and os.path.exists(oracle_ckpt):
        oracle_clf.load_state_dict(torch.load(oracle_ckpt, map_location=dev))
        oracle_clf.eval()
    else:
        print("Training Pixel-Space Oracle MNIST Classifier...")
        oracle_clf.fit(x_pixels, y_labels, epochs=20, device=dev)
        torch.save(oracle_clf.state_dict(), oracle_ckpt)

    # 3. Build Latent Space (PCA or VAE)
    if args.embedding == "pca":
        embedder = PCALatentSpace(latent_dim=args.latent_dim).to(dev)
        embedder.fit(x_pixels)
        z_data = embedder.encode(x_pixels)
    else:
        embedder = MNISTVAE(latent_dim=args.latent_dim).to(dev)
        vae_ckpt = os.path.join(ckpt_dir, f"mnist_vae_d{args.latent_dim}.pt")
        if use_cache and os.path.exists(vae_ckpt):
            print(f"[Cache] Loading MNIST VAE (D={args.latent_dim}) from {vae_ckpt}...")
            embedder.load_state_dict(torch.load(vae_ckpt, map_location=dev))
            embedder.eval()
        else:
            print(f"Training MNIST VAE (D={args.latent_dim})...")
            embedder.fit(x_pixels, epochs=45, device=dev)
            torch.save(embedder.state_dict(), vae_ckpt)
        z_data = embedder.encode(x_pixels)

    forb_mask = y_labels == args.forbidden_digit
    safe_mask = ~forb_mask
    z_forb = z_data[forb_mask]
    z_safe = z_data[safe_mask]

    perm_s = torch.randperm(z_safe.shape[0], device=dev)
    oracle_half_mmd = compute_mmd_rbf(
        z_safe[perm_s[: len(perm_s) // 2]],
        z_safe[perm_s[len(perm_s) // 2 : 2 * (len(perm_s) // 2)]],
    )

    # 4. Initialize Riemannian Metric on the Latent Space R^d
    base_metric = None
    if eff_metric == "pullback":
        base_metric = VAEPullbackMetric(
            decoder_mean_fn=embedder.decode,
            latent_data=z_data,
            output_dim=784,
            lambda_reg=1e-2,
        ).to(dev)
    elif eff_metric == "cdc":
        base_metric = CDCMetric(
            dim=args.latent_dim, rank=args.latent_dim, hidden_dim=128, lambda_reg=1e-2
        ).to(dev)
        cdc_ckpt = os.path.join(ckpt_dir, f"cdc_{args.embedding}_d{args.latent_dim}.pt")
        if use_cache and os.path.exists(cdc_ckpt):
            print(f"[Cache] Loading CDC Metric from {cdc_ckpt}...")
            base_metric.load_state_dict(torch.load(cdc_ckpt, map_location=dev))
            base_metric.eval()
        else:
            print(
                f"Training CDC Metric in {args.embedding.upper()} Latent Space (D={args.latent_dim})..."
            )
            base_metric.fit(z_data, epochs=600, device=dev)
            torch.save(base_metric.state_dict(), cdc_ckpt)
    elif eff_metric == "land":
        base_metric = LANDMetric(
            data=z_data, max_Reference_points=600, sigma=0.35, lambda_reg=1e-2
        ).to(dev)
    elif eff_metric == "rbf":
        base_metric = RBFMetric(data=z_data, num_centers=128, lambda_reg=1e-2).to(dev)

    # 5. Initialize Premetric
    premetric = None
    if args.premetric == "euclidean":
        premetric = EuclideanPremetric().to(dev)
    elif args.premetric in ["diffusion", "biharmonic"]:
        premetric = SpectralLaplacianPremetric(
            data=z_data, mode=args.premetric, num_landmarks=512
        ).to(dev)

    # 6. Construct Selected Forbidden Barrier Shape (`centroid`, `kmeans`, or `classifier`)
    if args.barrier == "centroid":
        c_forb = z_forb.mean(dim=0)
        dists_c = torch.linalg.norm(z_forb - c_forb.unsqueeze(0), dim=-1)
        r_min = torch.quantile(dists_c, 0.70).item()
        r_max = r_min * 1.55
        potential = CompactForbiddenPotential(
            center=c_forb, r_min=r_min, r_max=r_max, scale=35.0, premetric=premetric
        ).to(dev)

    elif args.barrier == "kmeans":
        potential = KMeansForbiddenPotential(
            forbidden_points=z_forb,
            num_clusters=args.num_clusters,
            r_min_quantile=0.75,
            buffer_ratio=1.60,
            decay_temp=6.0,
            scale=35.0,
            premetric=premetric,
        ).to(dev)

    elif args.barrier == "classifier":
        potential = ClassifierForbiddenPotential(
            dim=args.latent_dim,
            num_classes=10,
            forbidden_class=args.forbidden_digit,
            p_core=0.45,
            p_cutoff=0.06,
            scale=35.0,
        ).to(dev)
        clf_ckpt = os.path.join(
            ckpt_dir, f"latent_barrier_clf_{args.embedding}_d{args.latent_dim}.pt"
        )
        if use_cache and os.path.exists(clf_ckpt):
            print(f"[Cache] Loading Latent Barrier Classifier from {clf_ckpt}...")
            potential.classifier.load_state_dict(torch.load(clf_ckpt, map_location=dev))
            potential.classifier.eval()
            with torch.no_grad():
                if z_forb.shape[0] > 0:
                    potential.center.copy_(z_forb.mean(dim=0, keepdim=True))
        else:
            print("Training Latent MLP Barrier Classifier...")
            potential.fit_classifier(z_data, y_labels, epochs=200, device=dev)
            torch.save(potential.classifier.state_dict(), clf_ckpt)

    # 7. Configure Coupler & Train (or Load Cached) Flow Model
    active_metric = base_metric
    if is_baked and base_metric is not None and args.model == "rfm":
        active_metric = ConformalObstacleMetric(
            base_metric=base_metric, potential_fn=potential, beta=1.0
        ).to(dev)

    if args.ot == "none":
        coupler = IndependentCoupler()
    elif args.ot == "euclidean":
        coupler = MinibatchOTCoupler(
            premetric=EuclideanPremetric().to(dev), method=args.ot_method
        )
    else:
        coupler = MinibatchOTCoupler(premetric=premetric, method=args.ot_method)

    if args.model == "rfm":
        model = RiemannianFlowMatching(
            dim=args.latent_dim,
            metric=active_metric,
            premetric=premetric,
            coupler=coupler,
        ).to(dev)
        if use_cache and os.path.exists(flow_ckpt):
            print(f"[Cache] Loading trained RFM model from {flow_ckpt}...")
            model.v_net.load_state_dict(torch.load(flow_ckpt, map_location=dev))
            model.eval()
        else:
            model.fit(z_data, epochs=args.epochs, device=dev)
            torch.save(model.v_net.state_dict(), flow_ckpt)
            print(f"[Cache] Saved RFM model to {flow_ckpt}")

    elif args.model == "mfm":
        model = MetricFlowMatching(
            dim=args.latent_dim,
            metric=active_metric,
            potential=potential if is_baked else None,
            coupler=coupler,
        ).to(dev)

        # Stage 1: Load or Train Geodesic Interpolant
        if use_cache and os.path.exists(interp_ckpt):
            print(f"[Cache] Loading MFM Stage 1 Interpolant from {interp_ckpt}...")
            model.interpolant_net.load_state_dict(
                torch.load(interp_ckpt, map_location=dev)
            )
            model.interpolant_net.eval()
        else:
            model.fit_interpolant(z_data, epochs=int(args.epochs * 0.7), device=dev)
            torch.save(model.interpolant_net.state_dict(), interp_ckpt)
            print(f"[Cache] Saved MFM Stage 1 Interpolant to {interp_ckpt}")

        if base_metric is not None and args.premetric == "geodesic":
            geo_pm = MFMGeodesicPremetric(
                mfm_interpolant=model.interpolant_net,
                metric=base_metric,
                num_quad_steps=6,
            ).to(dev)
            if isinstance(potential, CompactForbiddenPotential):
                potential.premetric = geo_pm
            if args.ot == "premetric":
                model.coupler = MinibatchOTCoupler(
                    premetric=geo_pm, method=args.ot_method
                )

        # Stage 2: Load or Train Flow Velocity Network
        if use_cache and os.path.exists(flow_ckpt):
            print(f"[Cache] Loading MFM Stage 2 Velocity Network from {flow_ckpt}...")
            model.v_net.load_state_dict(torch.load(flow_ckpt, map_location=dev))
            model.eval()
        else:
            model.fit_flow(z_data, epochs=args.epochs, device=dev)
            torch.save(model.v_net.state_dict(), flow_ckpt)
            print(f"[Cache] Saved MFM Stage 2 Velocity Network to {flow_ckpt}")

    elif args.model == "rmeanfm":
        mfm_s1 = MetricFlowMatching(
            dim=args.latent_dim,
            metric=active_metric,
            potential=potential if is_baked else None,
            coupler=coupler,
        ).to(dev)

        # Stage 1 Interpolant is shared directly with MFM!
        if use_cache and os.path.exists(interp_ckpt):
            print(f"[Cache] Loading Stage 1 Interpolant from {interp_ckpt}...")
            mfm_s1.interpolant_net.load_state_dict(
                torch.load(interp_ckpt, map_location=dev)
            )
            mfm_s1.interpolant_net.eval()
        else:
            mfm_s1.fit_interpolant(z_data, epochs=int(args.epochs * 0.7), device=dev)
            torch.save(mfm_s1.interpolant_net.state_dict(), interp_ckpt)
            print(f"[Cache] Saved Stage 1 Interpolant to {interp_ckpt}")

        if base_metric is not None and args.premetric == "geodesic":
            geo_pm = MFMGeodesicPremetric(
                mfm_interpolant=mfm_s1.interpolant_net,
                metric=base_metric,
                num_quad_steps=6,
            ).to(dev)
            if isinstance(potential, CompactForbiddenPotential):
                potential.premetric = geo_pm
            if args.ot == "premetric":
                coupler = MinibatchOTCoupler(premetric=geo_pm, method=args.ot_method)

        model = RiemannianMeanFlow(
            dim=args.latent_dim,
            interpolant_net=mfm_s1.interpolant_net,
            metric=active_metric,
            coupler=coupler,
        ).to(dev)

        if use_cache and os.path.exists(flow_ckpt):
            print(f"[Cache] Loading RMeanFM Velocity Network from {flow_ckpt}...")
            model.u_net.load_state_dict(torch.load(flow_ckpt, map_location=dev))
            model.eval()
        else:
            model.fit(z_data, epochs=args.epochs, device=dev)
            torch.save(model.u_net.state_dict(), flow_ckpt)
            print(f"[Cache] Saved RMeanFM Velocity Network to {flow_ckpt}")

    # 8. Evaluate & Decode Generated Samples
    z0_eval = torch.randn(args.n_samples, args.latent_dim, device=dev)
    use_inf = args.obstacle_mode in ["inference", "both"]

    streamer = RiemannianRepulsionStreamer(
        flow_model=model,
        potential=potential if use_inf else None,
        metric=base_metric,
        alpha=args.alpha if use_inf else 0.0,
        gyro_curl=args.gyro_curl if use_inf else 0.0,
    )

    if args.model == "rmeanfm" and not use_inf:
        z1_gen, traj = model.sample(z0_eval, steps=5)
    else:
        z1_gen, traj = integrate_ode(streamer, z0_eval, steps=100, solver="rk4")

    with torch.no_grad():
        T_s, B_s, D_s = traj.shape
        viol_flat = potential.is_violated(traj.reshape(T_s * B_s, D_s)).reshape(
            T_s, B_s
        )
        latent_path_viol = viol_flat.any(dim=0).float().mean().item() * 100.0
        latent_end_viol = viol_flat[-1].float().mean().item() * 100.0

        decoded_imgs = embedder.decode(z1_gen)
        logits_oracle = oracle_clf(decoded_imgs)
        probs_oracle = torch.softmax(logits_oracle, dim=-1)
        pred_classes = probs_oracle.argmax(dim=-1)
        oracle_forb_rate = (
            pred_classes == args.forbidden_digit
        ).float().mean().item() * 100.0

        latent_safe_mmd = compute_mmd_rbf(z1_gen, z_safe)

    elapsed = time.perf_counter() - t_start
    print("-" * 85)
    print(
        f"[Latent Barrier] Path Viol: {latent_path_viol:.2f}% | Endpoint Viol: {latent_end_viol:.2f}%"
    )
    print(
        f"[Pixel Oracle]   Generated Digit '{args.forbidden_digit}' Rate: {oracle_forb_rate:.2f}% (Unsteered Baseline ~10.0%)"
    )
    print(
        f"[Manifold MMD]   Safe Latent MMD: {latent_safe_mmd:.5f} (Oracle Half-Split MMD: {oracle_half_mmd:.5f})"
    )
    print(f"[Runtime]        {elapsed:.1f}s")
    print("-" * 85)

    csv_path = os.path.join(args.results_dir, "mnist_summary.csv")
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if write_hdr:
            writer.writerow(
                [
                    "filename",
                    "embedding",
                    "latent_dim",
                    "forbidden_digit",
                    "barrier",
                    "model",
                    "metric",
                    "ot",
                    "obstacle_mode",
                    "latent_path_viol_pct",
                    "latent_end_viol_pct",
                    "pixel_oracle_forbid_pct",
                    "latent_safe_mmd",
                    "oracle_half_mmd",
                    "runtime_sec",
                ]
            )
        writer.writerow(
            [
                fname,
                args.embedding,
                args.latent_dim,
                args.forbidden_digit,
                args.barrier,
                args.model,
                eff_metric,
                ot_str,
                args.obstacle_mode,
                f"{latent_path_viol:.2f}",
                f"{latent_end_viol:.2f}",
                f"{oracle_forb_rate:.2f}",
                f"{latent_safe_mmd:.5f}",
                f"{oracle_half_mmd:.5f}",
                f"{elapsed:.1f}",
            ]
        )

    title_str = (
        f"{args.embedding.upper()}-d{args.latent_dim} | {args.barrier.upper()} | "
        f"Latent End Viol: {latent_end_viol:.1f}%, Pixel '{args.forbidden_digit}' Rate: {oracle_forb_rate:.1f}%"
    )
    plot_mnist_avoidance_results(
        latents=z_data,
        labels=y_labels,
        traj=traj,
        decoded_imgs=decoded_imgs,
        pred_classes=pred_classes,
        pred_probs=probs_oracle,
        potential=potential,
        forbidden_digit=args.forbidden_digit,
        title=title_str,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
