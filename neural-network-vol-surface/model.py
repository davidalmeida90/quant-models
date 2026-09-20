"""Render the plots for the neural-network-vol-surface page.

Pulls SPY option chain data (cached snapshot in vol_data/snapshots/), builds
the empirical IV surface, then trains two MLPs (small and big)
on the surface and renders prediction-vs-truth snapshots at several epoch
checkpoints.

Outputs PNGs into this folder:
  real_surface.png             — empirical SPY IV surface (the "truth")
  small_epoch_{N}.png          — small NN prediction at epoch N
  big_epoch_{N}.png            — big NN prediction at epoch N
  loss_curves.png              — train MSE per epoch for both nets
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
CHAIN_CSV = HERE / "data" / "spy_chain.csv"

# =============================================================================
# 1. Load the live SPY chain (pulled via yfinance, see fetch_chain.py)
# =============================================================================
df = pd.read_csv(CHAIN_CSV)
print(f"Loaded {len(df):,} rows from {CHAIN_CSV.name}")

# SPY spot recorded at fetch time (yfinance daily bar)
S0 = 741.75
print(f"Spot S0 = {S0:.2f}  (SPY close)")

# Keep only OTM side at each strike (the canonical IV surface)
df = df[((df["side"] == "call") & (df["K"] >= S0)) |
        ((df["side"] == "put")  & (df["K"] <= S0))]
# Median over duplicates at same (K, T)
gb = df.groupby(["K", "T"], as_index=False)["iv"].median()
print(f"Clean (K, T, IV) triplets: {len(gb):,}")

# Build a K/S × T grid via griddata (cubic, nearest fallback)
MN_LO, MN_HI = 0.85, 1.15
T_LO, T_HI   = 0.03, 0.40
N_K, N_T     = 40, 28
K_axis = np.linspace(MN_LO * S0, MN_HI * S0, N_K)
T_axis = np.linspace(T_LO, T_HI, N_T)
K_grid, T_grid = np.meshgrid(K_axis, T_axis)

pts = gb[["K", "T"]].values
vals = gb["iv"].values
SURFACE = griddata(pts, vals, (K_grid, T_grid), method="cubic")
SURFACE_NN = griddata(pts, vals, (K_grid, T_grid), method="nearest")
SURFACE = np.where(np.isnan(SURFACE), SURFACE_NN, SURFACE)
SURFACE = np.clip(SURFACE, 0.10, 0.45)
print(f"Built surface grid: {SURFACE.shape}, IV range "
      f"[{SURFACE.min():.3f}, {SURFACE.max():.3f}]")

Z_LO, Z_HI = 0.10, 0.45


# =============================================================================
# 2. Build features: five polynomial terms
# =============================================================================
def features(K, T, S0=S0):
    K = np.asarray(K).flatten()
    T = np.asarray(T).flatten()
    mn = K / S0
    return np.stack([mn, T, mn**2, T**2, mn * T], axis=1)


X_full = features(K_grid, T_grid)
y_full = SURFACE.flatten().reshape(-1, 1)

X_mean = X_full.mean(axis=0, keepdims=True)
X_std  = X_full.std(axis=0, keepdims=True) + 1e-9
X_norm = (X_full - X_mean) / X_std
Y_MEAN = float(y_full.mean())
y_centered = y_full - Y_MEAN


# =============================================================================
# 3. MLP (Adam, ReLU)
# =============================================================================
class MLP:
    def __init__(self, sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        self.mW, self.mb, self.vW, self.vb = [], [], [], []
        for i in range(len(sizes) - 1):
            std = math.sqrt(2.0 / sizes[i])
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * std)
            self.b.append(np.zeros((1, sizes[i + 1])))
            self.mW.append(np.zeros_like(self.W[-1]))
            self.mb.append(np.zeros_like(self.b[-1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))
        self.t = 0

    def forward(self, X):
        activations, z_values = [X], []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; z_values.append(z)
            a = np.maximum(0, z) if i < len(self.W) - 1 else z
            activations.append(a)
        return a, activations, z_values

    def backward(self, y, activations, z_values, lr,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        m = y.shape[0]
        delta = 2 * (activations[-1] - y) / m
        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            dW = a_prev.T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * (z_values[i - 1] > 0)
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * dW**2
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * db**2
            mW_hat = self.mW[i] / (1 - beta1**self.t)
            mb_hat = self.mb[i] / (1 - beta1**self.t)
            vW_hat = self.vW[i] / (1 - beta2**self.t)
            vb_hat = self.vb[i] / (1 - beta2**self.t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def count_params(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1]
               for i in range(len(layers) - 1))


SMALL_LAYERS = [5, 4, 4, 1]
BIG_LAYERS   = [5, 128, 128, 1]
EPOCHS       = 600
LR           = 0.02
CHECKPOINTS  = [0, 50, 200, 600]   # plot the surface at each


def train_with_snapshots(layers, label):
    mlp = MLP(layers, seed=42)
    losses = []
    snaps = {}
    for epoch in range(EPOCHS + 1):
        y_pred, activs, zs = mlp.forward(X_norm)
        loss = float(np.mean((y_pred - y_centered) ** 2))
        losses.append(loss)
        if epoch in CHECKPOINTS:
            snaps[epoch] = (y_pred.flatten() + Y_MEAN).reshape(SURFACE.shape)
        if epoch < EPOCHS:
            mlp.backward(y_centered, activs, zs, LR)
    print(f"  {label:>5}  final MSE = {losses[-1]:.5f}  RMSE = {math.sqrt(losses[-1])*100:.2f}%")
    return snaps, np.array(losses)


print(f"\nTraining SMALL MLP {SMALL_LAYERS} ({count_params(SMALL_LAYERS)} params)...")
SMALL_SNAPS, SMALL_LOSSES = train_with_snapshots(SMALL_LAYERS, "SMALL")
print(f"Training BIG   MLP {BIG_LAYERS} ({count_params(BIG_LAYERS)} params)...")
BIG_SNAPS, BIG_LOSSES = train_with_snapshots(BIG_LAYERS, "BIG")


# =============================================================================
# 4. Render PNGs (matplotlib 3D, navy color, gold accents to match site)
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlepad": 14,
})

NAVY   = "#191936"
ACCENT = "#323D90"
GOLD   = "#C9A24E"

def surface_plot(Z, title, filename, *,
                 truth=None, vmax_clip=0.45):
    """3D surface plot.  If truth given, overlay it as a wireframe for reference."""
    fig = plt.figure(figsize=(7.8, 5.0), dpi=130)
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_grid, T_grid, np.clip(Z, Z_LO, vmax_clip),
                            cmap=cm.viridis, vmin=Z_LO, vmax=vmax_clip,
                            edgecolor="none", alpha=0.92, antialiased=True)
    if truth is not None:
        ax.plot_wireframe(K_grid, T_grid, np.clip(truth, Z_LO, vmax_clip),
                          color="#FFFFFF", alpha=0.45, linewidth=0.6,
                          rstride=4, cstride=4)
    ax.set_xlabel("Strike K", labelpad=6, color=NAVY)
    ax.set_ylabel("T (years)", labelpad=6, color=NAVY)
    ax.set_zlabel("Implied σ", labelpad=6, color=NAVY)
    ax.set_zlim(Z_LO, vmax_clip)
    ax.view_init(elev=22, azim=-55)
    ax.set_title(title, color=NAVY, fontweight="bold", fontsize=12)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label("Implied σ", color=NAVY)
    fig.tight_layout()
    fig.savefig(CHARTS / filename, dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {filename}")


print("\nRendering plots...")
surface_plot(SURFACE, "SPY implied volatility surface (live yfinance chain)",
             "real_surface.png")

for ep in CHECKPOINTS:
    surface_plot(SMALL_SNAPS[ep],
                 f"SMALL net (5-4-4-1, 49 params) — epoch {ep}",
                 f"small_epoch_{ep:03d}.png", truth=SURFACE)
    surface_plot(BIG_SNAPS[ep],
                 f"BIG net (5-128-128-1, 17,409 params) — epoch {ep}",
                 f"big_epoch_{ep:03d}.png", truth=SURFACE)


# Training-loss curves (log Y so the 0.7%→10% structure is visible)
fig, ax = plt.subplots(figsize=(7.8, 3.4), dpi=130)
xs = np.arange(1, EPOCHS + 1)   # skip the wildly random epoch 0
ax.plot(xs, np.sqrt(SMALL_LOSSES[1:]) * 100, color="#C0392B",
        linewidth=2.2, label=f"SMALL (5-4-4-1, {count_params(SMALL_LAYERS)} params)")
ax.plot(xs, np.sqrt(BIG_LOSSES[1:]) * 100, color=ACCENT,
        linewidth=2.2, label=f"BIG (5-128-128-1, {count_params(BIG_LAYERS):,} params)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training RMSE (% IV)  —  log scale")
ax.set_yscale("log")
ax.set_title("Training error per epoch  —  capacity matters",
             color=NAVY, fontweight="bold")
ax.set_xlim(1, EPOCHS)
ax.grid(True, alpha=0.25, linestyle=":")
ax.legend(loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig(CHARTS / "loss_curves.png", dpi=130, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  wrote loss_curves.png")

print("\nAll plots written.")
