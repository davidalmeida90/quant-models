"""Generate all matplotlib images for the Gamma Surface scripts page.

Each plot mirrors a notebook cell on the published page. Run once locally:
    py -3 _make_plots.py
Outputs: gamma_vs_spot.png, gamma_vs_vol.png, gamma_vs_time.png,
         gamma_vs_strike.png, gamma_surface_T010.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).resolve().parent / "charts"
HERE.mkdir(parents=True, exist_ok=True)

# --- Black-Scholes gamma ---
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))

# Common defaults
S0, K0, T0, R0, SIG0 = 100.0, 100.0, 0.5, 0.05, 0.25

# Light, clean style used across the site's notebook plots
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":  "white",
    "axes.edgecolor":  "#cfd5e2",
    "axes.labelcolor": "#191936",
    "axes.titlecolor": "#191936",
    "axes.titlesize":  13,
    "axes.titleweight": "600",
    "axes.labelsize":  11,
    "axes.grid":       True,
    "grid.color":      "#eef1fa",
    "grid.linestyle":  "-",
    "grid.linewidth":  0.9,
    "xtick.color":     "#5b6178",
    "ytick.color":     "#5b6178",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family":     "DejaVu Sans",
    "legend.frameon":  False,
})

ACCENT = "#1E56B8"
STRIKE = "#E11D48"

# ---------- 1. Gamma vs Spot ----------
spots = np.linspace(60, 140, 400)
gammas = gamma(spots, K0, T0, R0, SIG0)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(spots, gammas, color=ACCENT, lw=2.2)
ax.axvline(K0, color=STRIKE, lw=1.2, ls="--", alpha=0.85, label=f"Strike K = {K0:.0f}")
ax.set_title(f"Gamma vs Spot price  (K={K0:.0f}, T={T0}, sigma={SIG0:.0%}, r={R0:.0%})")
ax.set_xlabel("Spot price S")
ax.set_ylabel("Gamma")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "gamma_vs_spot.png", dpi=140)
plt.close(fig)

# ---------- 2. Gamma vs Volatility ----------
vols = np.linspace(0.05, 1.20, 400)
gammas = gamma(S0, K0, T0, R0, vols)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(vols * 100, gammas, color=ACCENT, lw=2.2)
ax.set_title(f"Gamma vs Volatility  (ATM: S=K={K0:.0f}, T={T0}, r={R0:.0%})")
ax.set_xlabel("Volatility sigma (%)")
ax.set_ylabel("Gamma")
fig.tight_layout()
fig.savefig(HERE / "gamma_vs_vol.png", dpi=140)
plt.close(fig)

# ---------- 3. Gamma vs Time to expiry ----------
times = np.linspace(0.01, 2.0, 400)
gammas = gamma(S0, K0, times, R0, SIG0)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(times, gammas, color=ACCENT, lw=2.2)
ax.set_title(f"Gamma vs Time to expiry  (ATM: S=K={K0:.0f}, sigma={SIG0:.0%}, r={R0:.0%})")
ax.set_xlabel("Time to expiry T (years)")
ax.set_ylabel("Gamma")
ax.invert_xaxis()  # time flows toward expiry on the right
fig.tight_layout()
fig.savefig(HERE / "gamma_vs_time.png", dpi=140)
plt.close(fig)

# ---------- 4. Gamma vs Strike (with fixed spot) ----------
strikes = np.linspace(60, 140, 400)
gammas = gamma(S0, strikes, T0, R0, SIG0)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(strikes, gammas, color=ACCENT, lw=2.2)
ax.axvline(S0, color=STRIKE, lw=1.2, ls="--", alpha=0.85, label=f"Spot S = {S0:.0f}")
ax.set_title(f"Gamma vs Strike  (S={S0:.0f}, T={T0}, sigma={SIG0:.0%}, r={R0:.0%})")
ax.set_xlabel("Strike K")
ax.set_ylabel("Gamma")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "gamma_vs_strike.png", dpi=140)
plt.close(fig)

# ---------- 5. Gamma surface — T = 0.30 vs T = 0.10, SHARED z-axis ----------
moneyness = np.linspace(0.70, 1.30, 90)   # K/S
vols_grid = np.linspace(0.10, 0.80, 80)
M, V = np.meshgrid(moneyness, vols_grid)
K_grid = M * S0

# Compute both surfaces first so we can share zlim + colormap range
G_long  = gamma(S0, K_grid, 0.30, R0, V)
G_short = gamma(S0, K_grid, 0.10, R0, V)
Z_MAX   = max(G_long.max(), G_short.max()) * 1.02

for T_FIXED, G_surf, fname in [(0.30, G_long,  "gamma_surface_T030.png"),
                               (0.10, G_short, "gamma_surface_T010.png")]:
    fig = plt.figure(figsize=(9, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        M, V * 100, G_surf,
        cmap=cm.viridis,
        edgecolor="none",
        rstride=1, cstride=1,
        antialiased=True,
        alpha=0.95,
        vmin=0, vmax=Z_MAX,            # shared color scale
    )
    ax.set_zlim(0, Z_MAX)              # shared z-axis
    ax.set_title(f"Gamma surface at T = {T_FIXED:.2f}  (S={S0:.0f}, r={R0:.0%})")
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Volatility sigma (%)")
    ax.set_zlabel("Gamma")
    ax.view_init(elev=22, azim=-58)
    cb = fig.colorbar(surf, shrink=0.6, pad=0.10, label="Gamma")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=140)
    plt.close(fig)

# ---------- 6. Gamma-Theta MIRROR at T = 0.15, vs Strike ----------
def theta_call(S, K, T, r, sigma):
    dd1 = d1(S, K, T, r, sigma)
    dd2 = dd1 - sigma * np.sqrt(T)
    return (-S * norm.pdf(dd1) * sigma / (2.0 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(dd2))

T_MIRROR = 0.15
strikes_m = np.linspace(60, 140, 400)
g_mirror  = gamma(S0, strikes_m, T_MIRROR, R0, SIG0)
t_mirror  = theta_call(S0, strikes_m, T_MIRROR, R0, SIG0) / 365.0   # negative

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(strikes_m, g_mirror, color="#1E56B8", lw=2.4, label="Gamma  (positive — gains as spot moves)")
ax.plot(strikes_m, t_mirror, color="#E11D48", lw=2.4, label="Theta per day  (negative — daily bleed)")
ax.axhline(0, color="#9aa3b6", lw=0.8)
ax.axvline(S0, color="#9aa3b6", lw=0.8, ls="--", label=f"Spot S = {S0:.0f}")
ax.set_title(f"Gamma and Theta — mirror opposites at T = {T_MIRROR}  (sigma={SIG0:.0%}, r={R0:.0%})")
ax.set_xlabel("Strike K")
ax.set_ylabel("Value")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "gamma_theta_tradeoff.png", dpi=140)
plt.close(fig)

# ---------- 7. Zomma  =  dGamma / dSigma ----------
# Zomma = Gamma * (d1 * d2 - 1) / sigma
def zomma(S, K, T, r, sigma):
    dd1 = d1(S, K, T, r, sigma)
    dd2 = dd1 - sigma * np.sqrt(T)
    g   = gamma(S, K, T, r, sigma)
    return g * (dd1 * dd2 - 1.0) / sigma

T_3 = 0.15
strikes_3 = np.linspace(60, 140, 400)
g_3 = gamma(S0, strikes_3, T_3, R0, SIG0)
z_3 = zomma(S0, strikes_3, T_3, R0, SIG0)

fig, ax1 = plt.subplots(figsize=(9, 4.6))
ax1.plot(strikes_3, g_3, color="#1E56B8", lw=2.4, label="Gamma  (left axis)")
ax1.set_xlabel("Strike K")
ax1.set_ylabel("Gamma", color="#1E56B8")
ax1.tick_params(axis="y", labelcolor="#1E56B8")
ax1.axvline(S0, color="#9aa3b6", lw=0.8, ls="--")

ax2 = ax1.twinx()
ax2.plot(strikes_3, z_3, color="#16A34A", lw=2.4, label="Zomma  (right axis)")
ax2.axhline(0, color="#9aa3b6", lw=0.8)
ax2.set_ylabel("Zomma  (dGamma / dSigma)", color="#16A34A")
ax2.tick_params(axis="y", labelcolor="#16A34A")

ax1.set_title(f"Zomma vs Gamma at T = {T_3}  (S={S0:.0f}, sigma={SIG0:.0%}, r={R0:.0%})")
fig.tight_layout()
fig.savefig(HERE / "zomma.png", dpi=140)
plt.close(fig)

# ---------- 8. Vanna  =  d^2V / dS dSigma  =  -phi(d1) * d2 / sigma ----------
def vanna(S, K, T, r, sigma):
    dd1 = d1(S, K, T, r, sigma)
    dd2 = dd1 - sigma * np.sqrt(T)
    return -norm.pdf(dd1) * dd2 / sigma

v_3 = vanna(S0, strikes_3, T_3, R0, SIG0)

fig, ax1 = plt.subplots(figsize=(9, 4.6))
ax1.plot(strikes_3, g_3, color="#1E56B8", lw=2.4, label="Gamma  (left axis)")
ax1.set_xlabel("Strike K")
ax1.set_ylabel("Gamma", color="#1E56B8")
ax1.tick_params(axis="y", labelcolor="#1E56B8")
ax1.axvline(S0, color="#9aa3b6", lw=0.8, ls="--")

ax2 = ax1.twinx()
ax2.plot(strikes_3, v_3, color="#9333EA", lw=2.4, label="Vanna  (right axis)")
ax2.axhline(0, color="#9aa3b6", lw=0.8)
ax2.set_ylabel("Vanna  (dDelta / dSigma)", color="#9333EA")
ax2.tick_params(axis="y", labelcolor="#9333EA")

ax1.set_title(f"Vanna vs Gamma at T = {T_3}  (S={S0:.0f}, sigma={SIG0:.0%}, r={R0:.0%})")
fig.tight_layout()
fig.savefig(HERE / "vanna.png", dpi=140)
plt.close(fig)

print("Wrote 8 PNGs to", HERE)
