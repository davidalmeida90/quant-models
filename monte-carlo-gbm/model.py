"""Monte Carlo option pricing under geometric Brownian motion

Extracted from the published write-up at
https://davidariasfinance.com/scripts/monte-carlo-gbm/
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

# ── Parameters ──
S0    = 50            # Initial stock price
K     = 50            # Strike price
T     = 9 / 12        # Time to maturity (years)
r     = 0.04          # Risk-free interest rate
q     = 0.00          # Continuous dividend yield
sigma = 0.30          # Volatility (base)
N     = 10000000     # Number of simulations



np.random.seed(42)
z = np.random.normal(0, 1, N)

st = S0 * np.exp((r - q - 0.5 * sigma**2)  * T + sigma  * np.sqrt(T) * z)
st
np.shape(st)




# Terminal prices at two different implied volatilities
sigma_low  = 0.20
sigma_high = 0.30

S_T_low  = S0 * np.exp((r - q - 0.5 * sigma_low**2)  * T + sigma_low  * np.sqrt(T) * z)
S_T_high = S0 * np.exp((r - q - 0.5 * sigma_high**2) * T + sigma_high * np.sqrt(T) * z)

# Both share the same risk-neutral mean
mean_both  = np.mean(S_T_low)
median_low  = np.median(S_T_low)
median_high = np.median(S_T_high)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(S_T_low,  bins=100, density=True, alpha=0.55, color="steelblue",  label=f"σ = {sigma_low:.0%}")
ax.hist(S_T_high, bins=100, density=True, alpha=0.45, color="indianred", label=f"σ = {sigma_high:.0%}")


ax.axvline(mean_both, color="black", linestyle="-", linewidth=1.3, label=f"Mean ≈ {mean_both:.2f}")


ax.axvline(median_low,  color="steelblue", linestyle="--", linewidth=1.2, label=f"Median (σ=20%) = {median_low:.2f}")
ax.axvline(median_high, color="indianred", linestyle="--", linewidth=1.2, label=f"Median (σ=30%) = {median_high:.2f}")

ax.set_title("Terminal Price Distribution — σ = 20% vs 30%", fontweight="bold")
ax.set_xlabel("$S_T$")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()

print(median_low)

# Monte Carlo simulation (single-step terminal value)


# Call
payoff_call  = np.maximum(st - K, 0)
premium_call = np.mean(payoff_call) * np.exp(-r * T)
se_call      = np.std(payoff_call, ddof=1) / np.sqrt(N)

# Put
payoff_put  = np.maximum(K - st, 0)
premium_put = np.mean(payoff_put) * np.exp(-r * T)
se_put      = np.std(payoff_put, ddof=1) / np.sqrt(N)

payoff_call


#  Black-Scholes analytical prices (for comparison)
d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_call = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

print(premium_call, bs_call)
a = premium_call - bs_call
print(a)

## Paths matrix, now we go with two dimensions -> (n_paths, n_steps)
n_paths = 175
n_steps = int(T * 252)
dt = T / n_steps

np.random.seed(7)
Z = np.random.normal(0, 1, (n_paths, n_steps)) ## need to make a matrix of all z numbers


# Build price matrix: each row is one path
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = S0

np.shape(paths)

paths

# Paths composition and terminal value
for t in range(n_steps):
    paths[:, t + 1] = paths[:, t] * np.exp(
        (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t] ## its day by day, replace T by dt -- use a bidimensional array of z
    )

S_terminal = paths[:, -1]


#classification of terminal values, call option
atm_band = 0.5
is_itm = S_terminal > (K + atm_band)
is_otm = S_terminal < (K - atm_band)
is_atm = ~is_itm & ~is_otm

np.sum(is_atm)



days = np.arange(n_steps + 1)

plt.figure(figsize=(10, 5))

for i in np.where(is_otm)[0]:
    plt.plot(days, paths[i], color="red", alpha=0.15, linewidth=0.5)

for i in np.where(is_atm)[0]:
    plt.plot(days, paths[i], color="gold", alpha=0.6, linewidth=0.8)

for i in np.where(is_itm)[0]:
    plt.plot(days, paths[i], color="green", alpha=0.15, linewidth=0.5)

plt.axhline(K, color="black", linestyle="--", linewidth=1.2, label=f"Strike K = {K}")

plt.ylabel("Stock Price ($)")
plt.xlabel("Days")
plt.title(f"MC simulation — {n_paths} paths (σ = {sigma:.0%})")
plt.legend([
    plt.Line2D([], [], color="green",  linewidth=2),
    plt.Line2D([], [], color="gold",   linewidth=2),
    plt.Line2D([], [], color="red",    linewidth=2),
    plt.Line2D([], [], color="black",  linewidth=1.2, linestyle="--"),
], [
    f"ITM  ({np.sum(is_itm)})",
    f"ATM  ({np.sum(is_atm)})",
    f"OTM  ({np.sum(is_otm)})",
    f"Strike = {K}",
], loc="upper left")

plt.tight_layout()
plt.show()
