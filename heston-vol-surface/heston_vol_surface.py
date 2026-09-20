"""Heston volatility surface — hero + supporting 2D plots.

Renders the full visual set for the planned notebook page on
encore7finance.com (and the future IG/YT reel). Output goes to
  reel_generating_machine/output/heston_vol_surface/

Outputs:
  hero_static.png        — big cinematic 3D surface (the page hero)
  equation_panel.png     — Heston SDEs + parameter table
  smile_vs_rho.png       — 2D: skew shifts with rho
  smile_vs_xi.png        — 2D: smile curvature with vol-of-vol
  termstr_vs_kappa.png   — 2D: ATM term structure shifts with kappa
  termstr_vs_v0.png      — 2D: ATM term structure starting from different v_0
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

# ── IG-shape watermark raster (same construction as gamma_surface_reel) ──
def _build_ig_watermark(N: int = 256):
    img = np.zeros((N, N, 4), dtype=np.float32)
    ix_grid = np.arange(N); iy_grid = np.arange(N)
    IX, IY = np.meshgrid(ix_grid, iy_grid)
    t_grid = ((IX / (N - 1)) + (1 - IY / (N - 1))) / 2
    g_grid = 0.55 + 0.25 * t_grid
    img[..., 0] = g_grid; img[..., 1] = g_grid; img[..., 2] = g_grid
    img[..., 3] = 1.0
    cr = 0.22 * N
    dx = np.minimum(IX, N - 1 - IX); dy = np.minimum(IY, N - 1 - IY)
    corner = (dx < cr) & (dy < cr)
    d_corner = np.where(corner, np.sqrt((cr - dx) ** 2 + (cr - dy) ** 2), 0.0)
    outside = corner & (d_corner > cr)
    edge = corner & (d_corner > cr - 1.5) & (d_corner <= cr)
    img[outside, 3] = 0.0
    img[edge, 3] = (cr - d_corner[edge]) / 1.5
    cx = (N - 1) / 2; cy = (N - 1) / 2
    rx = np.abs(IX - cx); ry = np.abs(IY - cy)
    rad = np.sqrt((IX - cx) ** 2 + (IY - cy) ** 2)
    ring_outer_in  = N * 0.34
    ring_outer_out = ring_outer_in + N * 0.05
    inner_cr = N * 0.10
    in_outer = (rx <= ring_outer_out) & (ry <= ring_outer_out)
    in_inner = (rx <= ring_outer_in)  & (ry <= ring_outer_in)
    body_band = in_outer & ~in_inner
    body_corner = body_band & (rx > ring_outer_out - inner_cr) & (ry > ring_outer_out - inner_cr)
    d_bc = np.sqrt((rx - (ring_outer_out - inner_cr)) ** 2 + (ry - (ring_outer_out - inner_cr)) ** 2)
    body_corner_outside = body_corner & (d_bc > inner_cr)
    body_paint = body_band & ~body_corner_outside
    lens_r_in  = N * 0.18
    lens_r_out = N * 0.235
    lens_band = (rad > lens_r_in) & (rad <= lens_r_out)
    flash_cx = cx + N * 0.235; flash_cy = cy - N * 0.235; flash_r = N * 0.035
    flash_mask = np.sqrt((IX - flash_cx) ** 2 + (IY - flash_cy) ** 2) <= flash_r
    visible = img[..., 3] > 0
    paint = visible & (body_paint | lens_band | flash_mask)
    img[paint, 0:3] = 1.0
    return img

IG_WATERMARK = _build_ig_watermark()

def style_axes(ax, *, tick_color="#888888"):
    ax.set_facecolor(BG)
    ax.tick_params(colors=tick_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(True, color="#1A1A1A", linewidth=0.8)
    ax.set_axisbelow(True)

# =================================================================
#                              1. HERO 3D
# =================================================================
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

def render_hero():
    # Bumped density: 60 strikes x 50 maturities = 3000 cells (was 35x30 = 1050).
    # Smoother surface, matches gamma_surface_reel's render quality.
    strikes    = np.linspace(80, 120, 60)
    maturities = np.linspace(0.10, 1.0, 50)
    iv = _compute_iv_surface(strikes, maturities)

    K_mesh, T_mesh = np.meshgrid(strikes, maturities)
    # Same technique as gamma_surface_reel.py: explicit add_axes with the
    # 3D plot filling 95% of width and 85% of height, leaving the top 15%
    # for the corner labels at fig.text(y=0.93). Bypasses matplotlib's
    # default 3D auto-padding which leaves the surface looking small.
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax = fig.add_axes([0.025, 0.02, 0.95, 0.85], projection="3d", facecolor=BG)
    for pa in (ax.xaxis, ax.yaxis, ax.zaxis):
        pa.pane.set_edgecolor((0,0,0,0)); pa.pane.set_facecolor((0,0,0,0))
        pa.label.set_color(FG)
    ax.grid(False); ax.tick_params(colors="#666666", labelsize=9)
    ax.plot_surface(K_mesh, T_mesh, iv, cmap=CINEMATIC, edgecolor="none",
                    alpha=0.96, antialiased=True,
                    rcount=len(maturities), ccount=len(strikes), linewidth=0)
    atm = int(np.argmin(np.abs(strikes - S0_REF)))
    ax.plot(np.full_like(maturities, strikes[atm]), maturities, iv[:, atm],
            color="#FFE600", lw=3, zorder=10)
    ax.plot(strikes, np.full_like(strikes, maturities[0]), iv[0, :],
            color="#FF7A14", lw=3, zorder=10)
    ax.set_xlabel("Strike K", color=FG, fontsize=11, labelpad=8)
    ax.set_ylabel("Expiry T (years)", color=FG, fontsize=11, labelpad=8)
    ax.set_zlabel("Implied Vol", color=FG, fontsize=11, labelpad=8)
    ax.view_init(elev=14, azim=-62)
    # Push the 3D axes box to fill more of the figure — the surface itself
    # grows accordingly without changing where the corner fig.text sit.
    ax.set_position([0.0, 0.0, 1.0, 0.92])

    # No model title or parameter list — those live in the equation panel above.
    # Keep only the legend (which the equation panel doesn't carry) and the
    # current state readout (v_t, ATM IV).
    fig.text(0.02, 0.93, "YELLOW = ATM ridge    ORANGE = Front smile",
             color="#888888", fontsize=9, family="monospace")
    atm_iv_mid = iv[len(maturities)//2, atm] * 100
    fig.text(0.98, 0.93, f"v_t  = {V0_REF:.3f}", color=ACCENT, fontsize=12,
             family="monospace", weight="bold", ha="right")
    fig.text(0.98, 0.89, f"ATM IV = {atm_iv_mid:.1f}%", color=FG, fontsize=10,
             family="monospace", ha="right")
    # No tight_layout — the explicit add_axes positioning is the whole point.
    # No bbox_inches="tight" either — save the full figsize so the axes box
    # stays at 95% × 85% of the figure, not auto-shrunk to content.
    out = OUT_DIR / "hero_static.png"
    fig.savefig(out, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"[hero] Saved {out}  ({out.stat().st_size/1024:.0f} KB)")

# =================================================================
#                       2. EQUATION + INPUT PANEL
# =================================================================
def render_equation_panel():
    """Tightened layout: reduced vertical gaps between title, SDEs, corr, and params."""
    fig = plt.figure(figsize=(13, 3.6), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG); ax.axis("off")
    # Title
    ax.text(0.5, 0.93, "THE HESTON MODEL", color=ACCENT, fontsize=20,
            family="serif", weight="bold", ha="center", va="top")
    # The two SDEs (mathtext) - tighter spacing
    ax.text(0.5, 0.66,
            r"$dS_t \;=\; r\,S_t\,dt \;+\; \sqrt{v_t}\,S_t\,dW_t^{(1)}$",
            color=FG, fontsize=18, ha="center", va="center")
    ax.text(0.5, 0.46,
            r"$dv_t \;=\; \kappa(\theta - v_t)\,dt \;+\; \xi\,\sqrt{v_t}\,dW_t^{(2)}$",
            color=FG, fontsize=18, ha="center", va="center")
    ax.text(0.5, 0.27,
            r"$\mathrm{corr}\,(dW^{(1)},\, dW^{(2)}) \;=\; \rho$",
            color="#999999", fontsize=12, ha="center", va="center")
    # Parameter values
    ax.text(0.5, 0.08,
            f"kappa = {KAPPA_REF}     theta = {THETA_REF}     xi = {XI_REF}     "
            f"rho = {RHO_REF}     v_0 = {V0_REF}",
            color=ACCENT, fontsize=12, family="monospace",
            ha="center", va="center", weight="bold")
    out = OUT_DIR / "equation_panel.png"
    fig.savefig(out, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[equation] Saved {out}")

# =================================================================
#                  3. SUPPORTING 2D PLOTS (4 plots)
# =================================================================
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
def _plot_smile_vs_rho(ax):
    rhos = [-0.9, 0.0, 0.3]
    T = 0.25
    style_axes(ax)
    for rho, color in zip(rhos, TRIO_COLORS):
        K, iv = _smile_curve(T, rho=rho)
        ax.plot(K, iv*100, color=color, lw=2.4, label=f"rho = {rho:+.1f}")
    ax.set_xlabel("Strike K", color=FG, fontsize=10)
    ax.set_ylabel("Implied Vol (%)", color=FG, fontsize=10)
    ax.set_title(f"Effect of rho (skew)   T = {T}", color=ACCENT,
                 fontsize=12, weight="bold", pad=10)
    ax.legend(loc="upper center", facecolor=BG, edgecolor="#333333",
              labelcolor=FG, fontsize=8.5, ncol=3)

def _plot_smile_vs_xi(ax):
    xis = [0.2, 0.75, 1.5]
    T = 0.25
    style_axes(ax)
    for xi, color in zip(xis, TRIO_COLORS):
        K, iv = _smile_curve(T, xi=xi)
        ax.plot(K, iv*100, color=color, lw=2.4, label=f"xi = {xi}")
    ax.set_xlabel("Strike K", color=FG, fontsize=10)
    ax.set_ylabel("Implied Vol (%)", color=FG, fontsize=10)
    ax.set_title(f"Effect of xi (vol of vol)   T = {T}", color=ACCENT,
                 fontsize=12, weight="bold", pad=10)
    ax.legend(loc="upper center", facecolor=BG, edgecolor="#333333",
              labelcolor=FG, fontsize=8.5, ncol=3)

def _plot_termstr_vs_kappa(ax):
    kappas = [0.5, 2.0, 5.0]
    Ts = np.linspace(0.1, 2.0, 22)
    style_axes(ax)
    for kap, color in zip(kappas, TRIO_COLORS):
        iv = _atm_term_structure(Ts, kappa=kap)
        ax.plot(Ts, iv*100, color=color, lw=2.4, label=f"kappa = {kap}")
    ax.axhline(np.sqrt(THETA_REF)*100, color="#666666", ls="--", lw=1,
               label=f"sqrt(theta) = {np.sqrt(THETA_REF)*100:.1f}%")
    ax.set_xlabel("Expiry T (years)", color=FG, fontsize=10)
    ax.set_ylabel("ATM Implied Vol (%)", color=FG, fontsize=10)
    ax.set_title("Effect of kappa (mean-reversion speed)", color=ACCENT,
                 fontsize=12, weight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor="#333333",
              labelcolor=FG, fontsize=8.5, ncol=2)

def _plot_termstr_vs_v0(ax):
    v0s = [0.005, 0.05, 0.15]
    Ts = np.linspace(0.1, 2.0, 22)
    style_axes(ax)
    for v0, color in zip(v0s, TRIO_COLORS):
        iv = _atm_term_structure(Ts, v0=v0)
        ax.plot(Ts, iv*100, color=color, lw=2.4, label=f"v_0 = {v0}")
    ax.axhline(np.sqrt(THETA_REF)*100, color="#666666", ls="--", lw=1,
               label=f"sqrt(theta) = {np.sqrt(THETA_REF)*100:.1f}%")
    ax.set_xlabel("Expiry T (years)", color=FG, fontsize=10)
    ax.set_ylabel("ATM Implied Vol (%)", color=FG, fontsize=10)
    ax.set_title("Effect of v_0 (starting variance)", color=ACCENT,
                 fontsize=12, weight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor="#333333",
              labelcolor=FG, fontsize=8.5, ncol=2)

def render_param_grid_2x2():
    """Combine the 4 parameter-effect plots into one 2x2 grid image."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.24})
    _plot_smile_vs_rho(axes[0, 0])
    _plot_smile_vs_xi(axes[0, 1])
    _plot_termstr_vs_kappa(axes[1, 0])
    _plot_termstr_vs_v0(axes[1, 1])
    fig.suptitle("How each Heston parameter shapes the surface",
                 color=ACCENT, fontsize=16, family="serif", weight="bold",
                 y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = OUT_DIR / "param_grid_2x2.png"
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"[plot] Saved {out.name}  ({out.stat().st_size/1024:.0f} KB)")

# =================================================================
#         4. IG REEL STATIC — 1080x1920 vertical (single PNG)
# =================================================================
def render_reel_static_vertical():
    """All three blocks (title+eqs+params, 3D surface, 2x2 grid) in one
    1080x1920 vertical PNG ready for Instagram. figsize 10.8x19.2 @ dpi 100."""
    strikes    = np.linspace(80, 120, 60)
    maturities = np.linspace(0.10, 1.0, 50)
    iv = _compute_iv_surface(strikes, maturities)
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(10.8, 19.2), facecolor=BG, dpi=100)

    # ---- TITLE / EQUATIONS / PARAMS BLOCK ----
    # Net drag: -0.08 from original (was -0.10, then nudged back up by +0.02).
    # title 0.895 (orig 0.975) ... params 0.793 (orig 0.873).
    # Title in cyan-blue (matches BLUE_CINEMATIC surface palette; distinct from gamma's warm gold)
    fig.text(0.5, 0.895, "Heston Surface Simulation", color="#3CABE6", fontsize=26,
             family="serif", weight="bold", ha="center", va="top")
    fig.text(0.5, 0.863,
             r"$dS_t \;=\; r\,S_t\,dt \;+\; \sqrt{v_t}\,S_t\,dW_t^{(1)}$",
             color=FG, fontsize=17, ha="center", va="center")
    fig.text(0.5, 0.837,
             r"$dv_t \;=\; \kappa(\theta - v_t)\,dt \;+\; \xi\,\sqrt{v_t}\,dW_t^{(2)}$",
             color=FG, fontsize=17, ha="center", va="center")
    fig.text(0.5, 0.813,
             r"$\mathrm{corr}\,(dW^{(1)},\, dW^{(2)}) \;=\; \rho$",
             color="#999999", fontsize=11, ha="center", va="center")
    fig.text(0.5, 0.793,
             f"kappa = {KAPPA_REF}    theta = {THETA_REF}    xi = {XI_REF}    "
             f"rho = {RHO_REF}    v_0 = {V0_REF}",
             color=ACCENT, fontsize=11, family="monospace",
             ha="center", va="center", weight="bold")

    # ---- 3D SURFACE (middle, dominant block; box expanded both ways + camera zoomed in to grow the
    # actual surface render without moving the title/eq block above or the 2x2 subtitle/grid below.
    # Box: was [0.0, 0.372, 1.0, 0.488]; now top up by 0.012, bottom down by 0.006 -> height 0.506.
    # ax.dist lowered from default 10 to 7.5 to zoom the camera in by ~33% (works in mpl 3.6+).
    ax = fig.add_axes([0.0, 0.366, 1.0, 0.506], projection="3d", facecolor=BG)
    for pa in (ax.xaxis, ax.yaxis, ax.zaxis):
        pa.pane.set_edgecolor((0, 0, 0, 0)); pa.pane.set_facecolor((0, 0, 0, 0))
        pa.label.set_color(FG)
    ax.grid(False); ax.tick_params(colors="#666666", labelsize=8)
    # CINEMATIC_V2 palette (static preview only) — slight tweak of the warm CINEMATIC.
    ax.plot_surface(K_mesh, T_mesh, iv, cmap=CINEMATIC_V2, edgecolor="none",
                    alpha=0.96, antialiased=True,
                    rcount=len(maturities), ccount=len(strikes), linewidth=0)
    atm = int(np.argmin(np.abs(strikes - S0_REF)))
    ax.plot(np.full_like(maturities, strikes[atm]), maturities, iv[:, atm],
            color="#FFE600", lw=3, zorder=10)
    ax.plot(strikes, np.full_like(strikes, maturities[0]), iv[0, :],
            color="#FF7A14", lw=3, zorder=10)
    ax.set_xlabel("Strike K", color=FG, fontsize=10, labelpad=3)
    ax.set_ylabel("Expiry T", color=FG, fontsize=10, labelpad=3)
    ax.set_zlabel("Implied Vol", color=FG, fontsize=10, labelpad=3)
    # Camera elev 14 -> 22 (looks down more, less time spent looking at the surface base)
    ax.view_init(elev=22, azim=-62)
    ax.dist = 7.5
    # Explicit Z headroom — push max up 20% so the surface doesn't kiss the top of the box
    ax.set_zlim(float(iv.min()) * 0.85, float(iv.max()) * 1.20)

    # ---- Subtitle — bigger + extra-bold per request ----
    fig.text(0.5, 0.378, "Simulating v_t and rho",
             color="#3CABE6", fontsize=18, family="serif",
             weight="900",  # max bold weight
             ha="center", va="center")

    # ---- SIMULATE v_t SDE + rho sweep + ATM IV (mirrors animation v7's pre-compute) ----
    n_steps = 288
    T_min_sim, T_max_sim = 0.30, 1.50
    V_T_FLOOR_S = 0.015
    XI_SIM_S = 0.55  # was 0.45 — slightly more variance noise per user request
    FIXED_SEED = 7   # hardcoded — same trajectory every render (no more seed search)
    dt_sim = (T_max_sim - T_min_sim) / n_steps

    def _sim_path(seed_val):
        rng = np.random.default_rng(seed=seed_val)
        v = np.zeros(n_steps); v[0] = V0_REF
        for k in range(1, n_steps):
            v_prev = max(v[k-1], V_T_FLOOR_S)
            dW = rng.normal(0.0, np.sqrt(dt_sim))
            dv = KAPPA_REF * (THETA_REF - v_prev) * dt_sim + XI_SIM_S * np.sqrt(v_prev) * dW
            v[k] = max(v_prev + dv, V_T_FLOOR_S)
        return v

    v_t_seq = _sim_path(FIXED_SEED)
    # Smoothing (15-frame rolling avg) + seamless loop correction
    sw = 15; pad = sw // 2
    padded = np.concatenate([v_t_seq[-pad:], v_t_seq, v_t_seq[:pad]])
    v_t_seq = np.convolve(padded, np.ones(sw)/sw, mode='valid')
    v_t_seq = np.maximum(v_t_seq, V_T_FLOOR_S)
    correction = np.linspace(0.0, -(v_t_seq[-1] - v_t_seq[0]), n_steps)
    v_t_seq = np.maximum(v_t_seq + correction, V_T_FLOOR_S)
    # Rho sweep
    rho_seq = -0.6 + 0.3 * np.sin(2.0 * np.pi * np.arange(n_steps) / n_steps)
    # ATM IV per step (single point — fast, no full surface needed)
    T_atm_fixed = 0.5
    atm_iv_seq = np.zeros(n_steps)
    for k in range(n_steps):
        try:
            atm_iv_seq[k] = implied_vol(
                heston_call(S0_REF, T_atm_fixed, v0=v_t_seq[k], rho=rho_seq[k]),
                S0_REF, T_atm_fixed) * 100
        except Exception:
            atm_iv_seq[k] = atm_iv_seq[k-1] if k > 0 else np.nan

    # ---- 2 BOTTOM CHARTS (matches animation v7 layout) ----
    pos = [
        [0.080, 0.180, 0.395, 0.160],  # LEFT:  rho over time
        [0.525, 0.180, 0.395, 0.160],  # RIGHT: dual-y v_t + ATM IV
    ]
    # LEFT: rho over simulation step (gold line, full path traced)
    ax_l = fig.add_axes(pos[0]); style_axes(ax_l)
    ax_l.plot(np.arange(n_steps), rho_seq, color=ACCENT, lw=2.4)
    ax_l.set_xlabel("Simulation step", color=FG, fontsize=9)
    ax_l.set_ylabel("rho", color=ACCENT, fontsize=10, weight="bold")
    ax_l.tick_params(axis="y", colors=ACCENT, labelsize=8)
    ax_l.set_xlim(0, n_steps - 1); ax_l.set_ylim(-1.0, 0.0)
    ax_l.set_title("rho (correlation)", color="#3CABE6",
                   fontsize=12, weight="bold", pad=10)
    # RIGHT: dual-y v_t (gold, left) + ATM IV (cyan, right)
    ax_r = fig.add_axes(pos[1]); style_axes(ax_r)
    ax_r.plot(np.arange(n_steps), v_t_seq, color=ACCENT, lw=2.4)
    ax_r.set_xlabel("Simulation step", color=FG, fontsize=9)
    ax_r.set_ylabel("v_t", color=ACCENT, fontsize=10, weight="bold")
    ax_r.tick_params(axis="y", colors=ACCENT, labelsize=8)
    ax_r.set_xlim(0, n_steps - 1)
    ax_r.set_ylim(float(v_t_seq.min()) * 0.85, float(v_t_seq.max()) * 1.15)
    ax_r_iv = ax_r.twinx()
    ax_r_iv.set_facecolor((0, 0, 0, 0))
    for spine in ax_r_iv.spines.values(): spine.set_color("#333333")
    ax_r_iv.plot(np.arange(n_steps), atm_iv_seq, color="#00E5FF", lw=2.4)
    ax_r_iv.set_ylabel("ATM IV (%)", color="#00E5FF", fontsize=10, weight="bold")
    ax_r_iv.tick_params(axis="y", colors="#00E5FF", labelsize=8)
    ax_r_iv.set_ylim(float(np.nanmin(atm_iv_seq)) * 0.95, float(np.nanmax(atm_iv_seq)) * 1.05)
    ax_r.set_title("v_t & ATM IV evolution", color="#3CABE6",
                   fontsize=12, weight="bold", pad=10)

    # ---- IG-shape watermark + @davidarias_cfa handle, positioned top-LEFT next to the surface
    # (matches gamma_surface_reel placement at gy=0.69 — upper-left corner of the surface area).
    ax_wm = fig.add_axes([0, 0, 1, 1], zorder=99)
    ax_wm.set_facecolor((0, 0, 0, 0)); ax_wm.axis("off")
    ax_wm.set_xlim(0, 1); ax_wm.set_ylim(0, 1)
    wm_alpha_logo = 0.16
    wm_alpha_text = 0.32
    gw = 0.13  # bigger icon (was 0.10)
    gh = gw * (1080.0 / 1920.0)
    gx = 0.05
    gy = 0.69
    ax_wm.imshow(IG_WATERMARK, extent=(gx, gx + gw, gy, gy + gh),
                 aspect="auto", alpha=wm_alpha_logo, zorder=1,
                 transform=ax_wm.transAxes)
    ax_wm.text(gx + gw / 2, gy - 0.010, "@davidarias_cfa",
               transform=ax_wm.transAxes, fontsize=22, fontweight="800",
               color="#FFFFFF", ha="center", va="top",
               alpha=wm_alpha_text)

    out = OUT_DIR / "reel_static_1080x1920.png"
    fig.savefig(out, dpi=100, facecolor=BG)  # 10.8x19.2 @ 100 dpi = 1080x1920 exact
    plt.close(fig)
    print(f"[reel] Saved {out}  ({out.stat().st_size/1024:.0f} KB)  -> 1080x1920 vertical (BLUE preview)")

# =================================================================
#         5. 12-SEC ANIMATION — surface MORPHS as v_t evolves (IG reel)
# =================================================================
def render_animation_T_unfold(duration_sec=12, fps=24,
                              n_strikes=25, n_maturities=20,
                              T_min=0.30, T_max=1.50,
                              encode_mp4=True,
                              rotate_360=True):
    """Heston surface animation where the WHOLE SURFACE MORPHS over time (like
    gamma_surface_reel's morphing surface, not a moving slice). Mechanism:
       t (clock time)   sweeps T_min -> T_max  (linearly)
       v_t              evolves via the Heston variance SDE simulation
       IV surface       recomputed each frame with v_0 = v_t  -> surface shape transforms
    For speed, we pre-compute IV surfaces at ~30 sample v_t values, then linearly
    interpolate per frame. Layout matches render_reel_static_vertical exactly."""
    import os, subprocess, imageio_ffmpeg
    n_frames = int(duration_sec * fps)
    print(f"[anim] {n_frames} frames @ {fps}fps ({duration_sec}s) | IV grid {n_maturities}x{n_strikes}")

    strikes    = np.linspace(80, 120, n_strikes)
    maturities = np.linspace(0.12, 1.0, n_maturities)  # T_min = 0.12 (user request)

    # ---- t (clock) sequence: linear 0.3 -> 1.5 over 12 sec (just sweep index — no longer displayed) ----
    t_seq = np.linspace(T_min, T_max, n_frames)

    # ---- v_t SDE path: fixed seed (consistent trajectory across renders) ----
    V_T_FLOOR = 0.015
    XI_SIM = 0.55     # was 0.45 — slightly more noise per user request
    FIXED_SEED = 7    # hardcoded — no more seed search; same trajectory always
    dt_sim = (T_max - T_min) / n_frames

    def _simulate_path(seed_val):
        rng = np.random.default_rng(seed=seed_val)
        v = np.zeros(n_frames)
        v[0] = V0_REF
        for i in range(1, n_frames):
            v_prev = max(v[i-1], V_T_FLOOR)
            dW = rng.normal(0.0, np.sqrt(dt_sim))
            dv = KAPPA_REF * (THETA_REF - v_prev) * dt_sim + XI_SIM * np.sqrt(v_prev) * dW
            v[i] = max(v_prev + dv, V_T_FLOOR)
        return v

    print(f"[anim] Using fixed seed {FIXED_SEED} (XI_SIM={XI_SIM})...")
    v_t_seq = _simulate_path(FIXED_SEED)

    # ---- Rolling-average smoothing (15-frame window) to remove high-freq jitter ----
    smooth_window = 15
    pad = smooth_window // 2
    padded = np.concatenate([v_t_seq[-pad:], v_t_seq, v_t_seq[:pad]])
    v_t_seq = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode='valid')
    v_t_seq = np.maximum(v_t_seq, V_T_FLOOR)

    # ---- Seamless-loop correction: subtract a linear ramp so v_t[0] == v_t[-1] ----
    gap = v_t_seq[-1] - v_t_seq[0]
    correction = np.linspace(0.0, -gap, n_frames)
    v_t_seq = np.maximum(v_t_seq + correction, V_T_FLOOR)
    print(f"[anim] v_t path (XI_SIM={XI_SIM}, smoothed): range [{v_t_seq.min():.4f}, {v_t_seq.max():.4f}], "
          f"loop gap = {abs(v_t_seq[0]-v_t_seq[-1]):.6f}")

    # ---- rho slow sinusoidal sweep: -0.9 -> -0.3 -> -0.9 (one full cycle over n_frames) ----
    # Center -0.6, amplitude 0.3 -> stays negative (realistic leverage effect for equity vol)
    rho_seq = -0.6 + 0.3 * np.sin(2.0 * np.pi * np.arange(n_frames) / n_frames)
    print(f"[anim] rho sweep: range [{rho_seq.min():+.3f}, {rho_seq.max():+.3f}]")

    # ---- Pre-compute IV surfaces on a 2D (v_t x rho) grid; bilinearly interpolate per frame ----
    n_v_samples   = 15
    n_rho_samples = 8
    v_sample_min = max(v_t_seq.min() * 0.9, V_T_FLOOR)
    v_sample_max = v_t_seq.max() * 1.1
    v_t_samples = np.linspace(v_sample_min, v_sample_max, n_v_samples)
    rho_samples = np.linspace(rho_seq.min() - 0.05, rho_seq.max() + 0.05, n_rho_samples)
    total_samples = n_v_samples * n_rho_samples
    print(f"[anim] Pre-computing {n_v_samples} x {n_rho_samples} = {total_samples} sample IV surfaces "
          f"(grid {n_maturities}x{n_strikes})...")
    t0 = time.time()
    iv_samples_2d = np.zeros((n_v_samples, n_rho_samples, n_maturities, n_strikes))
    done = 0
    for vi in range(n_v_samples):
        for ri in range(n_rho_samples):
            for r, T in enumerate(maturities):
                for c, K in enumerate(strikes):
                    try:
                        iv_samples_2d[vi, ri, r, c] = implied_vol(
                            heston_call(K, T, v0=v_t_samples[vi], rho=rho_samples[ri]), K, T)
                    except Exception:
                        iv_samples_2d[vi, ri, r, c] = np.nan
            if np.isnan(iv_samples_2d[vi, ri]).any():
                from scipy.ndimage import distance_transform_edt
                idx = distance_transform_edt(np.isnan(iv_samples_2d[vi, ri]),
                                             return_distances=False, return_indices=True)
                iv_samples_2d[vi, ri] = iv_samples_2d[vi, ri][tuple(idx)]
            done += 1
        print(f"  [v_t {vi+1:2d}/{n_v_samples} done]  v_t={v_t_samples[vi]:.4f}  "
              f"({done}/{total_samples} surfaces, {time.time()-t0:.0f}s)")
    print(f"[anim] Sample surfaces done in {time.time()-t0:.0f}s")
    iv_z_max = float(iv_samples_2d.max() * 1.20)  # more headroom (was 1.05)
    iv_z_min = float(max(0.04, iv_samples_2d.min() * 0.85))

    # ---- Bilinear interpolation helper: looks up IV surface at any (v_t, rho) ----
    def _bilinear_iv(vt, rho_val):
        vi_hi = int(np.searchsorted(v_t_samples, vt))
        vi_hi = min(max(vi_hi, 1), n_v_samples - 1); vi_lo = vi_hi - 1
        av = (vt - v_t_samples[vi_lo]) / (v_t_samples[vi_hi] - v_t_samples[vi_lo])
        ri_hi = int(np.searchsorted(rho_samples, rho_val))
        ri_hi = min(max(ri_hi, 1), n_rho_samples - 1); ri_lo = ri_hi - 1
        ar = (rho_val - rho_samples[ri_lo]) / (rho_samples[ri_hi] - rho_samples[ri_lo])
        return ((1-av)*(1-ar) * iv_samples_2d[vi_lo, ri_lo]
              + (1-av)*ar     * iv_samples_2d[vi_lo, ri_hi]
              + av*(1-ar)     * iv_samples_2d[vi_hi, ri_lo]
              + av*ar         * iv_samples_2d[vi_hi, ri_hi])

    # ---- Frames dir ----
    frames_dir = OUT_DIR / "anim_frames_T_unfold"
    frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    atm = int(np.argmin(np.abs(strikes - S0_REF)))
    K_mesh_full, T_mesh_full = np.meshgrid(strikes, maturities)
    T_atm_idx = int(np.argmin(np.abs(maturities - 0.5)))

    # ---- Pre-compute ATM IV per frame (using bilinear interp over v_t x rho) ----
    print("[anim] Pre-computing ATM IV per frame...")
    atm_iv_seq = np.zeros(n_frames)
    for i in range(n_frames):
        iv_temp = _bilinear_iv(v_t_seq[i], rho_seq[i])
        atm_iv_seq[i] = iv_temp[T_atm_idx, atm] * 100

    # ---- Non-linear time mapping: FIRST 6 SEC GO FASTER ----
    # Ease-out: video frame i maps to simulation index `eased[i] * (n_frames-1)`.
    # Midpoint mapping with power 1.7 -> at frame n_frames/2 we're at ~69% of the simulation,
    # so the first 6 sec cover 69% of the SDE path and the last 6 sec cover the remaining 31%.
    # Loop endpoints preserved (sim_index[0]=0, sim_index[-1]=n_frames-1).
    progress = np.linspace(0.0, 1.0, n_frames)
    eased = 1.0 - (1.0 - progress) ** 1.7
    sim_indices = (eased * (n_frames - 1)).astype(int)
    v_t_seq    = v_t_seq[sim_indices]
    rho_seq    = rho_seq[sim_indices]
    atm_iv_seq = atm_iv_seq[sim_indices]
    print(f"[anim] ease-out time map applied: frame n/2 -> sim step {sim_indices[n_frames//2]} (of {n_frames-1})")

    # Fixed y-axis ranges for the dual-y chart, with a small padding
    vt_y_lo, vt_y_hi = float(v_t_seq.min()) * 0.85, float(v_t_seq.max()) * 1.15
    iv_y_lo, iv_y_hi = float(atm_iv_seq.min()) * 0.95, float(atm_iv_seq.max()) * 1.05
    print(f"[anim] dual-y axes: v_t [{vt_y_lo:.4f}, {vt_y_hi:.4f}]   ATM IV [{iv_y_lo:.1f}%, {iv_y_hi:.1f}%]")

    print(f"[anim] Rendering {n_frames} frames...")
    t_start = time.time()

    for i, t_now in enumerate(t_seq):
        v_t_now = v_t_seq[i]
        rho_now = rho_seq[i]
        # Bilinear interpolation: surface morphs in BOTH v_t (height) and rho (skew)
        iv_now = _bilinear_iv(v_t_now, rho_now)

        fig = plt.figure(figsize=(10.8, 19.2), facecolor=BG, dpi=100)

        # ---- TITLE / EQ / PARAMS BLOCK — exact positions from render_reel_static_vertical ----
        fig.text(0.5, 0.895, "Heston Surface Simulation", color=ACCENT, fontsize=26,
                 family="serif", weight="bold", ha="center", va="top")
        fig.text(0.5, 0.863,
                 r"$dS_t \;=\; r\,S_t\,dt \;+\; \sqrt{v_t}\,S_t\,dW_t^{(1)}$",
                 color=FG, fontsize=17, ha="center", va="center")
        fig.text(0.5, 0.837,
                 r"$dv_t \;=\; \kappa(\theta - v_t)\,dt \;+\; \xi\,\sqrt{v_t}\,dW_t^{(2)}$",
                 color=FG, fontsize=17, ha="center", va="center")
        fig.text(0.5, 0.813,
                 r"$\mathrm{corr}\,(dW^{(1)},\, dW^{(2)}) \;=\; \rho$",
                 color="#999999", fontsize=11, ha="center", va="center")
        fig.text(0.5, 0.793,
                 f"kappa = {KAPPA_REF}    theta = {THETA_REF}    xi = {XI_REF}    "
                 f"rho = {RHO_REF}    v_0 = {V0_REF}",
                 color=ACCENT, fontsize=11, family="monospace",
                 ha="center", va="center", weight="bold")
        # Dynamic 3-value readout — v_t, rho, ATM IV all moving
        atm_iv_now = iv_now[T_atm_idx, atm] * 100
        fig.text(0.5, 0.770,
                 f"Simulating v_t and rho    v_t = {v_t_now:.4f}    rho = {rho_now:+.3f}    ATM IV = {atm_iv_now:.1f}%",
                 color="#FFE600", fontsize=12, family="monospace",
                 ha="center", va="center", weight="bold")

        # ---- 3D SURFACE — same ax box as static. WHOLE SURFACE MORPHS as v_t evolves ----
        ax = fig.add_axes([0.0, 0.366, 1.0, 0.506], projection="3d", facecolor=BG)
        for pa in (ax.xaxis, ax.yaxis, ax.zaxis):
            pa.pane.set_edgecolor((0, 0, 0, 0)); pa.pane.set_facecolor((0, 0, 0, 0))
            pa.label.set_color(FG)
        ax.grid(False); ax.tick_params(colors="#666666", labelsize=8)
        ax.plot_surface(K_mesh_full, T_mesh_full, iv_now, cmap=CINEMATIC_V2, edgecolor="none",
                        alpha=0.96, antialiased=True,
                        rcount=n_maturities, ccount=n_strikes, linewidth=0)
        # Highlight curves on the morphing surface
        ax.plot(np.full_like(maturities, strikes[atm]), maturities, iv_now[:, atm],
                color="#FFE600", lw=3, zorder=10)
        ax.plot(strikes, np.full_like(strikes, maturities[0]), iv_now[0, :],
                color="#FF7A14", lw=3, zorder=10)
        ax.set_xlim(80, 120)
        ax.set_ylim(0.12, 1.0)  # match new maturities range
        ax.set_zlim(iv_z_min, iv_z_max)
        ax.set_xlabel("Strike K", color=FG, fontsize=10, labelpad=3)
        ax.set_ylabel("Expiry T", color=FG, fontsize=10, labelpad=3)
        ax.set_zlabel("Implied Vol", color=FG, fontsize=10, labelpad=3)
        # 360 rotation: azim sweeps a full circle over the entire animation duration.
        if rotate_360:
            azim_now = -62.0 - 360.0 * (i / max(n_frames - 1, 1))
        else:
            azim_now = -62.0
        ax.view_init(elev=22, azim=azim_now)  # raised camera (was elev=14)
        ax.dist = 7.5

        # ---- SUBTITLE — exact from static ----
        fig.text(0.5, 0.378, "How each Heston parameter shapes the surface",
                 color=ACCENT, fontsize=12, family="serif", weight="bold",
                 ha="center", va="center")

        # ---- 2 ANIMATED GRAPHS ----
        ax_l = fig.add_axes([0.080, 0.180, 0.395, 0.160])
        ax_r = fig.add_axes([0.525, 0.180, 0.395, 0.160])
        # LEFT: rho value over simulation step (gold line traced, moving dot at "now")
        style_axes(ax_l)
        idx_hist = np.arange(i + 1)
        ax_l.plot(idx_hist, rho_seq[:i+1], color=ACCENT, lw=2.4, label="rho")
        ax_l.scatter([i], [rho_seq[i]], color=ACCENT, s=40, zorder=20)
        ax_l.set_xlabel("Simulation step", color=FG, fontsize=9)
        ax_l.set_ylabel("rho", color=ACCENT, fontsize=10, weight="bold")
        ax_l.tick_params(axis="y", colors=ACCENT, labelsize=8)
        ax_l.set_xlim(0, n_frames - 1)
        ax_l.set_ylim(-1.0, 0.0)
        ax_l.set_title(f"rho (correlation)   current = {rho_now:+.3f}",
                       color=ACCENT, fontsize=12, weight="bold", pad=10)

        # RIGHT: dual-y line chart of v_t (gold, left axis) and ATM IV (cyan, right axis),
        # traced over the simulated path. Moving dots at "now". Fully dynamic — both
        # lines grow frame-by-frame, no static reference clutter.
        style_axes(ax_r)
        idx_hist = np.arange(i + 1)
        # v_t on the left y-axis (gold)
        ax_r.plot(idx_hist, v_t_seq[:i+1], color=ACCENT, lw=2.4, label="v_t")
        ax_r.scatter([i], [v_t_seq[i]], color=ACCENT, s=40, zorder=20)
        ax_r.set_xlabel("Simulation step", color=FG, fontsize=9)
        ax_r.set_ylabel("v_t", color=ACCENT, fontsize=10, weight="bold")
        ax_r.tick_params(axis="y", colors=ACCENT, labelsize=8)
        ax_r.set_xlim(0, n_frames - 1)
        ax_r.set_ylim(vt_y_lo, vt_y_hi)
        # ATM IV on a twin right y-axis (cyan)
        ax_r_iv = ax_r.twinx()
        ax_r_iv.set_facecolor((0, 0, 0, 0))
        for spine in ax_r_iv.spines.values():
            spine.set_color("#333333")
        ax_r_iv.plot(idx_hist, atm_iv_seq[:i+1], color="#00E5FF", lw=2.4, label="ATM IV")
        ax_r_iv.scatter([i], [atm_iv_seq[i]], color="#00E5FF", s=40, zorder=20)
        ax_r_iv.set_ylabel("ATM IV (%)", color="#00E5FF", fontsize=10, weight="bold")
        ax_r_iv.tick_params(axis="y", colors="#00E5FF", labelsize=8)
        ax_r_iv.set_ylim(iv_y_lo, iv_y_hi)
        ax_r.set_title("v_t & ATM IV evolution", color=ACCENT,
                       fontsize=12, weight="bold", pad=10)

        # ---- IG watermark + @davidarias_cfa handle (top-left, next to surface) ----
        ax_wm = fig.add_axes([0, 0, 1, 1], zorder=99)
        ax_wm.set_facecolor((0, 0, 0, 0)); ax_wm.axis("off")
        ax_wm.set_xlim(0, 1); ax_wm.set_ylim(0, 1)
        gw_wm = 0.13
        gh_wm = gw_wm * (1080.0 / 1920.0)
        gx_wm, gy_wm = 0.05, 0.69
        ax_wm.imshow(IG_WATERMARK, extent=(gx_wm, gx_wm + gw_wm, gy_wm, gy_wm + gh_wm),
                     aspect="auto", alpha=0.16, zorder=1,
                     transform=ax_wm.transAxes)
        ax_wm.text(gx_wm + gw_wm / 2, gy_wm - 0.010, "@davidarias_cfa",
                   transform=ax_wm.transAxes, fontsize=22, fontweight="800",
                   color="#FFFFFF", ha="center", va="top", alpha=0.32)

        out = frames_dir / f"frame_{i:04d}.png"
        fig.savefig(out, dpi=100, facecolor=BG)
        plt.close(fig)
        if (i + 1) % 24 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            print(f"  [{i+1:3d}/{n_frames}]  t={t_now:.2f}  v_t={v_t_now:.4f}  elapsed {elapsed:.0f}s  eta {eta:.0f}s")

    print(f"[anim] Frame render done in {time.time()-t_start:.0f}s")

    if not encode_mp4:
        print(f"[anim] Frames in {frames_dir}; skipping MP4 encode.")
        return

    # ---- ffmpeg encode (use imageio_ffmpeg's bundled binary) ----
    _ffdir = os.path.join(os.path.dirname(imageio_ffmpeg.__file__), "binaries")
    _ffbin = next(f for f in os.listdir(_ffdir)
                  if f.lower().startswith("ffmpeg") and f.lower().endswith(".exe"))
    ffmpeg_exe = os.path.join(_ffdir, _ffbin)
    mp4_out = OUT_DIR / f"heston_reel_T_unfold_{duration_sec}s_v8_2.mp4"  # v8.2 = v8.1 but y-axis labels back to gold (only chart titles stay blue)
    cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(mp4_out),
    ]
    print(f"[anim] Encoding MP4 -> {mp4_out.name}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[anim] MP4 saved: {mp4_out}  ({mp4_out.stat().st_size/1024/1024:.1f} MB)")

# =================================================================
#                              MAIN
# =================================================================
if __name__ == "__main__":
    print("=" * 60)
    render_equation_panel()
    print("=" * 60)
    render_hero()
    print("=" * 60)
    print("Rendering 4 parameter-effect plots in a single 2x2 grid...")
    t0 = time.time()
    render_param_grid_2x2()
    print(f"Grid done in {time.time()-t0:.1f}s.")
    print("=" * 60)
    print("Rendering 1080x1920 vertical IG reel static...")
    t0 = time.time()
    render_reel_static_vertical()
    print(f"Reel static done in {time.time()-t0:.1f}s.")
    print("=" * 60)
    print("All files in:", OUT_DIR)
