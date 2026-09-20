"""Heston stochastic volatility: characteristic function pricing, then the implied
volatility surface it produces. Write-up: https://davidariasfinance.com/scripts/heston-vol-surface/
"""
from __future__ import annotations
import os, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).parent
OUT_DIR = HERE / "output" / "heston_vol_surface"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====================== BASELINE PARAMETERS ======================
# Match @quant.traderr Frame A so the hero is faithful to the inspiration.
S0_REF       = 100.0
R_REF        = 0.03
Q_REF        = 0.0
V0_REF       = 0.019
KAPPA_REF    = 2.5
THETA_REF    = 0.07
XI_REF       = 0.75    # vol of vol (the @quant.traderr 'xi')
RHO_REF      = -0.72

# ====================== Heston pricing (parameterized) ======================
def heston_char(u, T, S0, r, q, v0, kappa, theta, xi, rho):
    """Albrecher 'little trap' form of Heston characteristic function."""
    u = np.asarray(u, dtype=complex)
    xi_h = kappa - rho * xi * 1j * u
    d = np.sqrt(xi_h**2 + xi**2 * (u * 1j + u**2))
    g2 = (xi_h - d) / (xi_h + d)
    C = (kappa * theta / xi**2) * (
        (xi_h - d) * T - 2.0 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2))
    )
    D = ((xi_h - d) / xi**2) * (1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T))
    return np.exp(C + D * v0 + 1j * u * (np.log(S0) + (r - q) * T))

def heston_call(K, T, *, S0=S0_REF, r=R_REF, q=Q_REF, v0=V0_REF,
                kappa=KAPPA_REF, theta=THETA_REF, xi=XI_REF, rho=RHO_REF):
    args = dict(S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    def p1(u):
        return np.real(np.exp(-1j * u * np.log(K)) * heston_char(u - 1j, T, **args) /
                       (1j * u * heston_char(-1j, T, **args)))
    def p2(u):
        return np.real(np.exp(-1j * u * np.log(K)) * heston_char(u, T, **args) / (1j * u))
    P1 = 0.5 + (1.0 / np.pi) * quad(p1, 1e-8, 100, limit=200)[0]
    P2 = 0.5 + (1.0 / np.pi) * quad(p2, 1e-8, 100, limit=200)[0]
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

def bs_call(K, T, sigma, S0=S0_REF, r=R_REF, q=Q_REF):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(price, K, T, S0=S0_REF, r=R_REF, q=Q_REF):
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: bs_call(K, T, s, S0, r, q) - price, 1e-4, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan

# ====================== STYLE ======================
plt.style.use("dark_background")
CINEMATIC = LinearSegmentedColormap.from_list("heston_cinematic", [
    (0.00, "#1B0440"), (0.18, "#3D0470"), (0.36, "#7A1968"),
    (0.55, "#C8344A"), (0.72, "#F5872A"), (0.88, "#FFCB36"),
    (1.00, "#FFEE7A"),
])
# Cool/blue cinematic palette — kept here in case we want to swap to it later.
BLUE_CINEMATIC = LinearSegmentedColormap.from_list("heston_blue_cinematic", [
    (0.00, "#03061A"),  # near-black navy
    (0.18, "#0A1A40"),  # deep navy
    (0.36, "#163B7A"),  # ocean
    (0.55, "#2870C0"),  # mid blue
    (0.72, "#3CABE6"),  # bright cyan-blue
    (0.88, "#7FDDF5"),  # pale cyan
    (1.00, "#D4F4FF"),  # ice
])
# Slight tweak of the ORIGINAL warm cinematic palette — less pink, closer to the source.
# Subtle differences: slightly deeper purple at the bottom, slightly more vibrant red mid,
# slightly warmer pale-gold highlight. Recognizable as the same warm family as CINEMATIC.
CINEMATIC_V2 = LinearSegmentedColormap.from_list("heston_cinematic_v2", [
    (0.00, "#100428"),  # slightly deeper purple-indigo
    (0.18, "#420A78"),  # richer royal purple
    (0.36, "#7E1865"),  # magenta-warm
    (0.55, "#CC3045"),  # punchier red
    (0.72, "#F88A28"),  # vibrant warm orange
    (0.88, "#FFC838"),  # rich gold
    (1.00, "#FFEC80"),  # warm pale gold
])
LINE_COLORS = ["#7A1968", "#C8344A", "#F5872A", "#FFCB36"]   # 4-line palette
TRIO_COLORS = ["#7A1968", "#F5872A", "#FFCB36"]              # 3-line palette (purple/orange/gold, max contrast on black)
BG = "#000000"
FG = "#E0E0E0"
ACCENT = "#FFCB36"


def _compute_iv_surface(strikes, maturities):
    """Compute IV surface; defensively backfill NaN by nearest neighbor."""
    print(f"[iv-surface] Computing {len(maturities)}x{len(strikes)} = {len(maturities)*len(strikes)} cells...")
    t0 = time.time()
    iv = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            try:
                iv[i, j] = implied_vol(heston_call(K, T), K, T)
            except Exception:
                iv[i, j] = np.nan
    if np.isnan(iv).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(iv), return_distances=False, return_indices=True)
        iv = iv[tuple(idx)]
    print(f"[iv-surface] Done in {time.time()-t0:.1f}s")
    return iv


def _smile_curve(T, **overrides):
    """Compute IV smile across strikes at one T, with optional param overrides.
    Tighter range (85-115) so deep-OTM brentq failures don't show as wing artifacts."""
    strikes = np.linspace(85, 115, 25)
    ivs = np.full_like(strikes, np.nan, dtype=float)
    for j, K in enumerate(strikes):
        try:
            ivs[j] = implied_vol(heston_call(K, T, **overrides), K, T,
                                 S0=overrides.get("S0", S0_REF),
                                 r=overrides.get("r", R_REF),
                                 q=overrides.get("q", Q_REF))
        except Exception:
            pass
    return strikes, ivs


def _atm_term_structure(maturities, **overrides):
    """Compute ATM IV across maturities, with optional param overrides."""
    ivs = np.full_like(maturities, np.nan, dtype=float)
    for i, T in enumerate(maturities):
        try:
            ivs[i] = implied_vol(heston_call(S0_REF, T, **overrides), S0_REF, T,
                                 S0=overrides.get("S0", S0_REF),
                                 r=overrides.get("r", R_REF),
                                 q=overrides.get("q", Q_REF))
        except Exception:
            pass
    return ivs

# Each plot function accepts an optional `ax`. If None, it renders as its own
# standalone PNG. If passed, it plots into the provided axes (used by the
# 2x2 grid composite).
