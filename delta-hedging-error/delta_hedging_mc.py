"""Delta hedging error, a Monte Carlo on gamma.

Sell an at-the-money call, hedge it once a day with Black-Scholes delta, and measure what one
day of hedging earns or loses. Prices follow geometric Brownian motion with realised volatility
equal to implied and drift at the risk-free rate, so the hedge is right on average and every cent of
error comes from gamma meeting a discrete rebalance. Shares are bought with borrowed cash, and that
cash pays interest every day.

    python delta_hedging_mc.py

Writes figures/*.png and results.json next to this file. Needs numpy, scipy and matplotlib.
Runs in under a minute on a laptop.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- 1. parameters
SIGMA = 0.18            # implied vol, and realised vol, annualised
S0 = K = 100.0          # the call starts exactly at the money
R = 0.02                # flat risk-free rate: prices drift at it and the hedge's cash pays it
DT = 1 / 252            # one trading day
LIFE_DAYS = 20          # the option is sold 20 trading days before expiry
N_PATHS = 150_000
SEED = 11


# --------------------------------------------------------------------------- 2. Black-Scholes
def d1(S, K, T, sig):
    return (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))


def call_price(S, K, T, sig):
    a = d1(S, K, T, sig)
    return S * norm.cdf(a) - K * np.exp(-R * T) * norm.cdf(a - sig * np.sqrt(T))


def call_delta(S, K, T, sig):
    return norm.cdf(d1(S, K, T, sig))


def gamma(S, K, T, sig):
    return norm.pdf(d1(S, K, T, sig)) / (S * sig * np.sqrt(T))


# ------------------------------------------------------------------ 3. simulated price paths
rng = np.random.default_rng(SEED)
Z1 = rng.standard_normal(N_PATHS)      # where spot has wandered to by the start of the day
Z2 = rng.standard_normal(N_PATHS)      # the move during that one day


def one_day_of_hedging(tau_days):
    """P&L of a short call hedged with delta shares over one day, starting tau_days before expiry.

    Z1 and Z2 are reused for every tau, common random numbers, so results at different
    maturities differ because of time and nothing else. Buying delta shares uses more cash than
    the premium brings in, so the cash balance is negative and pays interest over the day.
    """
    elapsed = (LIFE_DAYS - tau_days) * DT
    s = S0 * np.exp((R - 0.5 * SIGMA ** 2) * elapsed + SIGMA * np.sqrt(elapsed) * Z1)
    s_next = s * np.exp((R - 0.5 * SIGMA ** 2) * DT + SIGMA * np.sqrt(DT) * Z2)
    t0, t1 = tau_days * DT, (tau_days - 1) * DT
    c0 = call_price(s, K, t0, SIGMA)
    c1 = np.maximum(s_next - K, 0.0) if t1 <= 1e-9 else call_price(s_next, K, t1, SIGMA)
    hedge = call_delta(s, K, t0, SIGMA)
    cash = c0 - hedge * s                              # premium in, shares bought, negative
    pnl = -(c1 - c0) + hedge * (s_next - s) + cash * np.expm1(R * DT)
    return s / K, pnl


# ------------------------------------------------------------ 4. spread of P&L by moneyness
EDGES = np.linspace(0.94, 1.06, 49)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
AT_STRIKE = int(np.argmin(np.abs(CENTRES - 1.0)))


def spread_by_moneyness(m, pnl, min_count=900):
    idx = np.digitize(m, EDGES) - 1
    ok = (idx >= 0) & (idx < len(CENTRES))
    n = np.bincount(idx[ok], minlength=len(CENTRES)).astype(float)
    s1 = np.bincount(idx[ok], weights=pnl[ok], minlength=len(CENTRES))
    s2 = np.bincount(idx[ok], weights=pnl[ok] ** 2, minlength=len(CENTRES))
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    std[n < min_count] = np.nan
    return std


# ------------------------------------------------------------------ 5. hedging more often
def last_day_rebalanced(n_rebalances, paths=200_000, seed=3):
    """Final day only, spot opening exactly at the strike, delta rebalanced n times in the day."""
    g = np.random.default_rng(seed)
    h = DT / n_rebalances
    s = np.full(paths, K)
    cash = np.full(paths, float(call_price(K, K, DT, SIGMA)))     # premium received
    shares = np.zeros(paths)
    for i in range(n_rebalances):
        hedge = call_delta(s, K, DT - i * h, SIGMA)
        cash -= (hedge - shares) * s                               # buy or sell the difference
        shares = hedge
        s = s * np.exp((R - 0.5 * SIGMA ** 2) * h + SIGMA * np.sqrt(h) * g.standard_normal(paths))
        cash *= np.exp(R * h)                                      # interest until the next rebalance
    pnl = cash + shares * s - np.maximum(s - K, 0.0)               # close the book at expiry
    return float(pnl.std()), float(pnl.mean())


# ---------------------------------------------------------------------------- 6. run it all
results = {"params": {"sigma": SIGMA, "S0": S0, "K": K, "r": R, "life_days": LIFE_DAYS,
                      "paths": N_PATHS, "seed": SEED}}

checks = []
for tau in (10, 5, 2, 1):
    m, pnl = one_day_of_hedging(tau)
    near = np.abs(m - 1) < 0.002
    value = float(call_price(K, K, tau * DT, SIGMA))
    after = float(call_price(K, K, (tau - 1) * DT, SIGMA)) if tau > 1 else 0.0
    delta0 = float(call_delta(K, K, tau * DT, SIGMA))
    one_day_decay = value - after + (value - delta0 * K) * float(np.expm1(R * DT))   # spot unchanged
    spread = float(spread_by_moneyness(m, pnl)[AT_STRIKE])
    checks.append({
        "T_days": tau,
        "option_value": value,
        "gamma": float(gamma(K, K, tau * DT, SIGMA)),
        "delta": delta0,
        "spread_at_strike": spread,
        "error_share_of_value": spread / value,
        "best_day_simulated": float(pnl[near].max()),
        "best_day_theory": one_day_decay,
        "worst_day": float(pnl.min()),
        "mean_pnl": float(pnl.mean()),
        "wings_spread": float(pnl[np.abs(m - 1) > 0.05].std()),
    })
results["by_maturity"] = checks

rebal = []
for n in (1, 2, 4, 12, 26, 78):
    sd, mean = last_day_rebalanced(n)
    rebal.append({"rebalances": n, "every_minutes": 390 / n, "spread": sd, "mean": mean})
results["rebalancing"] = rebal

(HERE / "results.json").write_text(json.dumps(results, indent=2))

# ------------------------------------------------------------------------------- 7. figures
INK, MUTED, GRID = "#191936", "#6b7480", "#e6e8ee"
NAVY, TEAL, AMBER, CRIMSON = "#323D90", "#1F7A8C", "#C58A1B", "#B3243F"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 15, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "font.size": 11.5,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=100)
    plt.close(fig)


# the P&L cloud, far from expiry and on the last day
def cloud(figsize, name):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    dots = np.random.default_rng(1).choice(N_PATHS, 14_000, replace=False)
    lim = None
    for ax, tau in zip(axes, (10, 1)):
        m, pnl = one_day_of_hedging(tau)
        lim = lim or float(np.percentile(np.abs(one_day_of_hedging(1)[1]), 99.5))
        ax.scatter(m[dots], pnl[dots], s=4, color=NAVY, alpha=0.16, edgecolors="none")
        ax.axvline(1.0, color=AMBER, lw=1.2, ls=":")
        ax.axhline(0.0, color=MUTED, lw=0.8)
        ax.set_xlim(0.94, 1.06)
        ax.set_ylim(-lim * 1.3, lim * 0.75)
        ax.set_title(f"T = {tau} days" if tau > 1 else "T = 1 day, the last day")
        ax.set_xlabel("spot / strike at the start of the day")
    axes[0].set_ylabel("hedge P&L over that day, $")
    save(fig, name)


cloud((11, 5.2), "pnl_cloud.png")
cloud((12, 6.3), "og_image.png")

# spread by moneyness
fig, ax = plt.subplots(figsize=(10, 5.4))
for tau, col in ((10, TEAL), (5, AMBER), (1, CRIMSON)):
    ax.plot(CENTRES, spread_by_moneyness(*one_day_of_hedging(tau)), color=col, lw=2.2,
            label=f"T = {tau} day" + ("s" if tau > 1 else ""))
ax.axvline(1.0, color=MUTED, lw=1.0, ls=":")
ax.set_xlim(0.94, 1.06)
ax.set_ylim(0, None)
ax.set_xlabel("spot / strike at the start of the day")
ax.set_ylabel("spread of daily hedge P&L, $")
ax.set_title("Spread of one day's hedging error, by moneyness")
ax.legend(frameon=False)
save(fig, "spread_by_moneyness.png")

# what drives it, at the strike
grid_T = np.linspace(10, 1, 541)
coarse = np.linspace(10, 1, 37)
err_coarse = np.array([spread_by_moneyness(*one_day_of_hedging(t))[AT_STRIKE] for t in coarse])
err = np.interp(grid_T[::-1], coarse[::-1], err_coarse[::-1])[::-1]
g_line = gamma(K, K, grid_T * DT, SIGMA)
v_line = call_price(K, K, grid_T * DT, SIGMA)
share = err / v_line
fig, ax = plt.subplots(figsize=(10, 5.4))
for y, col, lab in ((g_line / g_line[0], TEAL, "gamma"),
                    (v_line / v_line[0], AMBER, "option value"),
                    (share / share[0], CRIMSON, "hedge error / option value")):
    ax.plot(grid_T, y, color=col, lw=2.4, label=lab)
ax.set_yscale("log")
ax.set_xlim(10, 1)
ax.set_yticks([0.3, 1, 3, 10])
ax.set_yticklabels(["0.3x", "1x", "3x", "10x"])
ax.axhline(1.0, color=MUTED, lw=0.9)
ax.set_xlabel("T, trading days to expiry")
ax.set_ylabel("multiple of its level at T = 10")
ax.set_title("At the strike: gamma rises, option value falls, the error share explodes")
ax.legend(frameon=False, loc="upper left")
save(fig, "drivers_at_strike.png")

# hedging more often on the last day
ns = np.array([r["rebalances"] for r in rebal])
sds = np.array([r["spread"] for r in rebal])
fig, ax = plt.subplots(figsize=(10, 5.4))
fine = np.geomspace(1, 78, 100)
ax.plot(fine, sds[0] / np.sqrt(fine), color=MUTED, lw=1.4, ls="--", label="1 / square root of N")
ax.plot(ns, sds, "o", color=CRIMSON, ms=8, label="simulated")
ax.axhline(checks[0]["spread_at_strike"], color=NAVY, lw=1.2, ls=":",
           label="T = 10 days, hedged once a day")
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("rebalances during the last day")
ax.set_ylabel("spread of last day hedge P&L, $")
ax.set_title("Last day, spot opening at the strike: rebalancing more often")
ax.legend(frameon=False)
save(fig, "rebalancing.png")

print(json.dumps(results, indent=2))
