"""Cox-Ross-Rubinstein binomial tree, European and American

Extracted from the published write-up at
https://davidariasfinance.com/scripts/crr-binomial-tree/
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def crr_parameters(sigma, n, r, q, T):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    return u, d, p


def crr_stock(sigma, n, r, q, S, K, T):
    dt = T / n
    u, d, p = crr_parameters(sigma, n, r, q, T)
    smat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            smat[j, i] = S * u ** (i - j) * d ** j
    return smat

S     = 100.0    # Spot price
K     = 100.0    # Strike price (ATM)
T     = 1.0      # Time to expiry (years)
r     = 0.05     # Risk-free rate
q     = 0.0      # Dividend yield
sigma = 0.20     # Volatility
n     = 4        # Number of time steps

u, d, p = crr_parameters(sigma, n, r, q, T)
print(f"dt = {T/n:.4f}")
print(f"u  = {u:.6f}")
print(f"d  = {d:.6f}")
print(f"p  = {p:.6f}")

smat = crr_stock(sigma, n, r, q, S, K, T)
print(f"\nStock price tree:\n{np.round(smat, 2)}")

def crr_european(sigma, n, r, q, S, K, T, option_type='call'):
    dt = T / n
    omat = np.zeros((n + 1, n + 1))
    u, d, p = crr_parameters(sigma, n, r, q, T)
    smat = crr_stock(sigma, n, r, q, S, K, T)

    # STEP 1: PAYOFFS at MATURITY
    for j in range(0, n + 1):
        if option_type == 'call':
            omat[j, -1] = max(smat[j, -1] - K, 0)
        if option_type == 'put':
            omat[j, -1] = max(K - smat[j, -1], 0)

    # STEP 2: BACKWARDS INDUCTION
    for i in range(n - 1, -1, -1):
        for j in range(0, i + 1):
            omat[j, i] = (p * omat[j, i + 1] + (1 - p) * omat[j + 1, i + 1]) * np.exp(-r * dt)

    return omat

omat = crr_european(sigma, n, r, q, S, K, T, option_type='call')
print(f"European Call Price: ${omat[0, 0]:.4f}")

from scipy.stats import norm

def bs_price(S, K, T, r, q, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

bs = bs_price(S, K, T, r, q, sigma, option_type='call')
crr = omat[0, 0]
diff = crr - bs

print(f"Black-Scholes Price : ${bs:.4f}")
print(f"CRR European (n={n})  : ${crr:.4f}")
print(f"Difference          : ${diff:.4f}")

def crr_american(sigma, n, r, q, S, K, T, option_type='call'):
    dt = T / n
    amat = np.zeros((n + 1, n + 1))
    u, d, p = crr_parameters(sigma, n, r, q, T)
    smat = crr_stock(sigma, n, r, q, S, K, T)

    # STEP 1: PAYOFFS at MATURITY
    for j in range(0, n + 1):
        if option_type == 'call':
            amat[j, -1] = max(smat[j, -1] - K, 0)
        if option_type == 'put':
            amat[j, -1] = max(K - smat[j, -1], 0)

    # STEP 2: BACKWARDS INDUCTION with EARLY EXERCISE
    for i in range(n - 1, -1, -1):
        for j in range(0, i + 1):
            if option_type == 'call':
                amat[j, i] = max(
                    (p * amat[j, i + 1] + (1 - p) * amat[j + 1, i + 1]) * np.exp(-r * dt),
                    max(smat[j, i] - K, 0)
                )
            if option_type == 'put':
                amat[j, i] = max(
                    (p * amat[j, i + 1] + (1 - p) * amat[j + 1, i + 1]) * np.exp(-r * dt),
                    max(K - smat[j, i], 0)
                )
    return amat

amat = crr_american(sigma, n, r, q, S, K, T, option_type='call')
print(f"American Call Price: ${amat[0, 0]:.4f}")

def plot_crr_tree(sigma, n, r, q, S, K, T, option_type='call'):
    dt = T / n
    u, d, p = crr_parameters(sigma, n, r, q, T)
    smat = crr_stock(sigma, n, r, q, S, K, T)
    amat = crr_american(sigma, n, r, q, S, K, T, option_type)

    fig, ax = plt.subplots(1, 1, figsize=(4 * (n + 1), 3 * (n + 1)))

    for i in range(n + 1):
        for j in range(i + 1):
            x = i
            y = i - 2 * j  # vertical position: top = all ups, bottom = all downs

            stock = smat[j, i]
            option_val = amat[j, i]

            if option_type == 'call':
                intrinsic = max(stock - K, 0)
            else:
                intrinsic = max(K - stock, 0)

            # Determine continuation value (at maturity, continuation = intrinsic)
            if i < n:
                continuation = (p * amat[j, i + 1] + (1 - p) * amat[j + 1, i + 1]) * np.exp(-r * dt)
                early_exercise = intrinsic > continuation + 1e-10 and intrinsic > 0
            else:
                early_exercise = False

            color = '#FF6B6B' if early_exercise else '#4ECDC4'
            edge_color = '#C0392B' if early_exercise else '#2C3E50'

            bbox = dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor=edge_color, linewidth=2)
            label = f"S={stock:.2f}\nC={option_val:.2f}\nS-K={stock - K:.2f}"
            ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', bbox=bbox)

            # Draw edges to children
            if i < n:
                x_child = i + 1
                y_up = (i + 1) - 2 * j
                y_down = (i + 1) - 2 * (j + 1)
                ax.annotate('', xy=(x_child - 0.3, y_up), xytext=(x + 0.3, y),
                            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
                ax.annotate('', xy=(x_child - 0.3, y_down), xytext=(x + 0.3, y),
                            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

    ax.set_xlim(-0.8, n + 0.8)
    ax.set_ylim(-(n + 1.5), n + 1.5)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels([f"t={i}" for i in range(n + 1)], fontsize=10)
    ax.set_yticks([])
    ax.set_title(f"CRR American {option_type.title()} Tree  (S={S}, K={K}, T={T}, n={n}, σ={sigma}, r={r})",
                 fontsize=14, fontweight='bold')

    normal_patch = mpatches.Patch(facecolor='#4ECDC4', edgecolor='#2C3E50', label='Continuation optimal')
    exercise_patch = mpatches.Patch(facecolor='#FF6B6B', edgecolor='#C0392B', label='Early exercise optimal')
    ax.legend(handles=[normal_patch, exercise_patch], loc='upper left', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.show()


plot_crr_tree(sigma, n, r, q, S, K, T, option_type='call')
