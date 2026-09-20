"""Render all the static images for the encore7finance Heston notebook page.

Outputs go directly into:
    encore7finance/scripts/heston-vol-surface/

Files produced:
    heston_smile_vs_rho.png        — IV smile vs strike for 3 rho values
    heston_smile_vs_xi.png         — IV smile vs strike for 3 xi values
    heston_termstr_vs_kappa.png    — ATM IV term structure for 3 kappas
    heston_termstr_vs_v0.png       — ATM IV term structure for 3 v_0s
    heston_smile_across_T.png      — IV smile at T = 0.1, 0.5, 1.0
    heston_surface_v0_low.png      — full IV surface at v_0 = 0.015 (low vol regime)
    heston_surface_v0_high.png     — full IV surface at v_0 = 0.090 (high vol regime)
"""
from __future__ import annotations
import os, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from heston_vol_surface import (
    heston_call, implied_vol,
    S0_REF, R_REF, Q_REF, V0_REF, KAPPA_REF, THETA_REF, XI_REF, RHO_REF,
    _smile_curve, _atm_term_structure,
)

OUT_DIR = Path(__file__).resolve().parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Light theme matching gamma notebook page (white background, viridis surface).
plt.style.use("default")
LINE_COLORS = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444"]  # indigo, amber, emerald, red


def _save(fig, name: str, dpi: int = 140):
    out = OUT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {name}  ({out.stat().st_size/1024:.0f} KB)")


def plot_smile_vs_rho():
    rhos = [-0.9, -0.5, 0.0, 0.3]
    T = 0.30
    strikes = np.linspace(80, 120, 35)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    print(f"[smile_vs_rho] T = {T}")
    for rho, color in zip(rhos, LINE_COLORS):
        Ks, ivs = _smile_curve(T, rho=rho)
        # _smile_curve uses K in [85, 115] by default; widen
        ivs_arr = np.full_like(strikes, np.nan)
        for j, K in enumerate(strikes):
            try:
                ivs_arr[j] = implied_vol(heston_call(K, T, rho=rho), K, T)
            except Exception:
                pass
        ax.plot(strikes, ivs_arr * 100, color=color, lw=2.4, label=f"rho = {rho:+.1f}")
    ax.axvline(S0_REF, color="gray", ls="--", lw=0.8, alpha=0.7, label=f"Spot K = {S0_REF:.0f}")
    ax.set_title(f"Heston smile vs Strike — effect of rho   (T = {T}, v_0 = {V0_REF}, xi = {XI_REF})")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "heston_smile_vs_rho.png")


def plot_smile_vs_xi():
    xis = [0.2, 0.5, 1.0, 1.5]
    T = 0.30
    strikes = np.linspace(80, 120, 35)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    print(f"[smile_vs_xi] T = {T}")
    for xi, color in zip(xis, LINE_COLORS):
        ivs_arr = np.full_like(strikes, np.nan)
        for j, K in enumerate(strikes):
            try:
                ivs_arr[j] = implied_vol(heston_call(K, T, xi=xi), K, T)
            except Exception:
                pass
        ax.plot(strikes, ivs_arr * 100, color=color, lw=2.4, label=f"xi = {xi}")
    ax.axvline(S0_REF, color="gray", ls="--", lw=0.8, alpha=0.7, label=f"Spot K = {S0_REF:.0f}")
    ax.set_title(f"Heston smile vs Strike — effect of xi (vol of vol)   (T = {T}, rho = {RHO_REF})")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "heston_smile_vs_xi.png")


def plot_termstr_vs_kappa():
    kappas = [0.5, 1.5, 3.0, 5.0]
    Ts = np.linspace(0.1, 2.0, 24)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    print(f"[termstr_vs_kappa]")
    for kappa, color in zip(kappas, LINE_COLORS):
        ivs = _atm_term_structure(Ts, kappa=kappa)
        ax.plot(Ts, ivs * 100, color=color, lw=2.4, label=f"kappa = {kappa}")
    ax.axhline(np.sqrt(THETA_REF) * 100, color="gray", ls="--", lw=1.0,
               label=f"sqrt(theta) = {np.sqrt(THETA_REF)*100:.1f}%")
    ax.set_title(f"ATM term structure — effect of kappa (mean-reversion speed)   (v_0 = {V0_REF}, theta = {THETA_REF})")
    ax.set_xlabel("Time to expiry T (years)"); ax.set_ylabel("ATM Implied Vol (%)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "heston_termstr_vs_kappa.png")


def plot_termstr_vs_v0():
    v0s = [0.005, 0.02, 0.07, 0.15]
    Ts = np.linspace(0.1, 2.0, 24)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    print(f"[termstr_vs_v0]")
    for v0, color in zip(v0s, LINE_COLORS):
        ivs = _atm_term_structure(Ts, v0=v0)
        ax.plot(Ts, ivs * 100, color=color, lw=2.4, label=f"v_0 = {v0}")
    ax.axhline(np.sqrt(THETA_REF) * 100, color="gray", ls="--", lw=1.0,
               label=f"sqrt(theta) = {np.sqrt(THETA_REF)*100:.1f}%")
    ax.set_title(f"ATM term structure — effect of v_0 (starting variance)   (kappa = {KAPPA_REF}, theta = {THETA_REF})")
    ax.set_xlabel("Time to expiry T (years)"); ax.set_ylabel("ATM Implied Vol (%)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "heston_termstr_vs_v0.png")


def plot_smile_across_T():
    Ts_plot = [0.10, 0.50, 1.00]
    strikes = np.linspace(80, 120, 35)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    print(f"[smile_across_T]")
    for T_val, color in zip(Ts_plot, LINE_COLORS):
        ivs_arr = np.full_like(strikes, np.nan)
        for j, K in enumerate(strikes):
            try:
                ivs_arr[j] = implied_vol(heston_call(K, T_val), K, T_val)
            except Exception:
                pass
        ax.plot(strikes, ivs_arr * 100, color=color, lw=2.4, label=f"T = {T_val}")
    ax.axvline(S0_REF, color="gray", ls="--", lw=0.8, alpha=0.7, label=f"Spot K = {S0_REF:.0f}")
    ax.set_title(f"Heston smile flattens as T grows   (default Heston params)")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "heston_smile_across_T.png")


def _compute_full_iv_surface(strikes, maturities, **overrides):
    iv = np.zeros((len(maturities), len(strikes)))
    for r, T in enumerate(maturities):
        for c, K in enumerate(strikes):
            try:
                iv[r, c] = implied_vol(heston_call(K, T, **overrides), K, T)
            except Exception:
                iv[r, c] = np.nan
    if np.isnan(iv).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(iv), return_distances=False, return_indices=True)
        iv = iv[tuple(idx)]
    return iv


def plot_surface_comparison():
    """Two side-by-side surface snapshots at low vs high v_0 — shows surface morphing."""
    strikes    = np.linspace(80, 120, 50)
    maturities = np.linspace(0.12, 1.0, 40)
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)

    print("[surface_low_v0] computing...")
    t0 = time.time()
    iv_low  = _compute_full_iv_surface(strikes, maturities, v0=0.015)
    print(f"  done {time.time()-t0:.0f}s")
    print("[surface_high_v0] computing...")
    t0 = time.time()
    iv_high = _compute_full_iv_surface(strikes, maturities, v0=0.090)
    print(f"  done {time.time()-t0:.0f}s")

    # Shared z-axis for fair comparison
    z_max = float(max(iv_low.max(), iv_high.max())) * 1.05
    z_min = float(min(iv_low.min(), iv_high.min())) * 0.85

    for tag, iv, title in [
        ("low",  iv_low,  "Heston IV surface — low vol regime (v_0 = 0.015)"),
        ("high", iv_high, "Heston IV surface — high vol regime (v_0 = 0.090)"),
    ]:
        fig = plt.figure(figsize=(9, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(K_mesh, T_mesh, iv, cmap=cm.viridis, edgecolor="none",
                        alpha=0.95, vmin=z_min, vmax=z_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(title)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Expiry T (years)")
        ax.set_zlabel("Implied Vol")
        ax.view_init(elev=22, azim=-58)
        fig.tight_layout()
        _save(fig, f"heston_surface_v0_{tag}.png")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Rendering Heston notebook-page images...")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 60)
    plot_smile_vs_rho()
    plot_smile_vs_xi()
    plot_termstr_vs_kappa()
    plot_termstr_vs_v0()
    plot_smile_across_T()
    plot_surface_comparison()
    print("=" * 60)
    print(f"All assets done in {time.time()-t0:.0f}s")
    print(f"Files in: {OUT_DIR}")
