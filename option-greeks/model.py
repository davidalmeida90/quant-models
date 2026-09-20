"""Option Greeks: delta, gamma, theta, vega, rho

Extracted from the published write-up at
https://davidariasfinance.com/scripts/option-greeks/
Run: py -3 model.py
"""

# --- chart output ------------------------------------------------------------
# Notebook code calls plt.show(); here every figure is written to charts/ instead.
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
from pathlib import Path as _Path
_CHARTS = _Path(__file__).resolve().parent / "charts"
_CHARTS.mkdir(parents=True, exist_ok=True)
_fig_n = [0]
def _save_figure(*_a, **_k):
    _fig_n[0] += 1
    _plt.gcf().savefig(_CHARTS / f"figure_{_fig_n[0]:02d}.png", dpi=140, bbox_inches="tight", facecolor="white")
    _plt.close("all")
_plt.show = _save_figure
# -----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

S = 100.0      # Spot price
K = 100.0      # Strike
T = 1.0        # Time to expiry, in years
r = 0.05       # Risk-free rate
sigma = 0.25   # Volatility

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def delta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d_1)
    return norm.cdf(d_1) - 1.0

print(f"Call delta: {delta(S, K, T, r, sigma, 'call'):.4f}")
print(f"Put delta : {delta(S, K, T, r, sigma, 'put'):.4f}")

def gamma(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))

print(f"Gamma: {gamma(S, K, T, r, sigma):.4f}")

def theta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    first = -S * norm.pdf(d_1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return first - r * K * np.exp(-r * T) * norm.cdf(d_2)
    return first + r * K * np.exp(-r * T) * norm.cdf(-d_2)

print(f"Call theta (per year): {theta(S, K, T, r, sigma, 'call'):.4f}")
print(f"Call theta (per day) : {theta(S, K, T, r, sigma, 'call') / 365:.4f}")

def vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d_1) / 100.0

print(f"Vega (per 1pp of vol): {vega(S, K, T, r, sigma):.4f}")

spots = np.linspace(60, 140, 200)
deltas  = [delta(s, K, T, r, sigma, "call") for s in spots]
gammas  = [gamma(s, K, T, r, sigma) for s in spots]
thetas  = [theta(s, K, T, r, sigma, "call") / 365 for s in spots]  # per day
vegas   = [vega(s, K, T, r, sigma) for s in spots]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].plot(spots, deltas);  axes[0, 0].set_title("Delta (call)")
axes[0, 1].plot(spots, gammas);  axes[0, 1].set_title("Gamma")
axes[1, 0].plot(spots, thetas);  axes[1, 0].set_title("Theta per day (call)")
axes[1, 1].plot(spots, vegas);   axes[1, 1].set_title("Vega per 1pp")

for ax in axes.flat:
    ax.axvline(K, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Spot price")

plt.tight_layout()
plt.show()

spots = np.linspace(60, 140, 400)
maturities = [1.0, 0.5, 0.25, 0.10, 0.02]

plt.figure(figsize=(9, 5))
for T_i in maturities:
    g = [gamma(s, K, T_i, r, sigma) for s in spots]
    plt.plot(spots, g, label=f"T = {T_i:.2f}")

plt.axvline(K, color="red", linestyle="--", alpha=0.5)
plt.title("Gamma vs Spot Price for several expiries")
plt.xlabel("Spot price")
plt.ylabel("Gamma")
plt.legend()
plt.show()
