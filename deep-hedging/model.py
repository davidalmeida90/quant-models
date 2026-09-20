"""Render the plots for the deep-hedging page.

Focus: the hedge CURVE (delta), a neural network that LEARNS the hedge, and
TAIL RISK (CVaR). No backtest.

Outputs PNGs into this folder:
  hedge_vs_price.png   - hedge delta vs price S, at three volatility levels
  hedge_vs_vol.png     - hedge delta vs volatility, at three price levels
  hedge_surface.png    - single-call delta as a 3D surface of (price, volatility)
  spread_surface.png   - a 98/102 call-spread hedge as a 3D ridge
  nn_loss.png          - training error of a small MLP learning the spread hedge
  nn_fit.png           - the MLP's learned surface vs the true spread hedge
  tail_risk.png        - terminal P&L distribution + 99% VaR / CVaR
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).resolve().parent / "charts"
HERE.mkdir(parents=True, exist_ok=True)
NAVY, ACCENT, GOLD = "#191936", "#323D90", "#C9A24E"
GREEN, RED, BLUE = "#1E9E5A", "#C0392B", "#2E6FE0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlepad": 12})

def ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))

def call_delta(S, vol, tau, K=100.0):
    S = np.asarray(S, float); vol = np.asarray(vol, float)
    d1 = (np.log(S / K) + 0.5 * vol * vol * tau) / (vol * np.sqrt(tau))
    return ncdf(d1)

K, TAU = 100.0, 15 / 250

# ---------------------------------------------------------------------------
# 1. Hedge vs PRICE
# ---------------------------------------------------------------------------
S = np.linspace(88, 112, 240)
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=130)
for vol, col in [(0.15, BLUE), (0.25, ACCENT), (0.35, RED)]:
    ax.plot(S, call_delta(S, vol, TAU), color=col, lw=2.4, label=f"σ = {vol:.0%}")
ax.axvline(K, color="#888", lw=1.0, ls=":"); ax.text(K + 0.4, 0.05, "K = 100", color="#666", fontsize=9)
ax.set_xlabel("Price  S", color=NAVY); ax.set_ylabel("hedge  δ = N(d₁)", color=NAVY)
ax.set_title("Hedge vs price: the delta curve (slope = gamma)", color=NAVY, fontweight="bold", fontsize=12)
ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.25, linestyle=":")
ax.legend(loc="upper left", frameon=True, title="15 days to expiry")
fig.tight_layout(); fig.savefig(HERE / "hedge_vs_price.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote hedge_vs_price.png")

# ---------------------------------------------------------------------------
# 2. Hedge vs VOLATILITY
# ---------------------------------------------------------------------------
V = np.linspace(0.05, 0.55, 240)
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=130)
for S0, lab, col in [(96.0, "OTM (S = 96)", BLUE), (100.0, "ATM (S = 100)", ACCENT), (104.0, "ITM (S = 104)", RED)]:
    ax.plot(V, call_delta(S0, V, TAU), color=col, lw=2.4, label=lab)
ax.axhline(0.5, color="#888", lw=1.0, ls=":"); ax.text(0.06, 0.52, "δ = 0.5", color="#666", fontsize=9)
ax.set_xlabel("Volatility  σ", color=NAVY); ax.set_ylabel("hedge  δ = N(d₁)", color=NAVY)
ax.set_title("Hedge vs volatility: σ pulls delta toward ½ (vanna)", color=NAVY, fontweight="bold", fontsize=12)
ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.25, linestyle=":")
ax.legend(loc="center right", frameon=True, title="K = 100, 15 days")
fig.tight_layout(); fig.savefig(HERE / "hedge_vs_vol.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote hedge_vs_vol.png")

# ---------------------------------------------------------------------------
# 3. Single-call delta SURFACE
# ---------------------------------------------------------------------------
S_ax = np.linspace(88, 112, 60); V_ax = np.linspace(0.08, 0.50, 60)
SS, VV = np.meshgrid(S_ax, V_ax)
DELTA = call_delta(SS, VV, TAU)
fig = plt.figure(figsize=(7.8, 5.0), dpi=130); ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(SS, VV, DELTA, cmap=cm.coolwarm, edgecolor="none", alpha=0.95, antialiased=True)
ax.set_xlabel("Price  S", labelpad=6, color=NAVY); ax.set_ylabel("Volatility  σ", labelpad=6, color=NAVY)
ax.set_zlabel("hedge  δ", labelpad=6, color=NAVY); ax.view_init(elev=24, azim=-58)
ax.set_title("The hedge is a surface of price and volatility", color=NAVY, fontweight="bold", fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08).set_label("hedge δ", color=NAVY)
fig.tight_layout(); fig.savefig(HERE / "hedge_surface.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote hedge_surface.png")

# ---------------------------------------------------------------------------
# 4. CALL-SPREAD hedge surface (a ridge) — the position from the video
# ---------------------------------------------------------------------------
KA, KB = 98.0, 102.0
def spread_delta(S, vol, tau): return call_delta(S, vol, tau, KA) - call_delta(S, vol, tau, KB)
SPREAD = spread_delta(SS, VV, TAU)
fig = plt.figure(figsize=(7.8, 5.0), dpi=130); ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(SS, VV, SPREAD, cmap=cm.coolwarm, edgecolor="none", alpha=0.95, antialiased=True)
ax.set_xlabel("Price  S", labelpad=6, color=NAVY); ax.set_ylabel("Volatility  σ", labelpad=6, color=NAVY)
ax.set_zlabel("hedge  δ", labelpad=6, color=NAVY); ax.view_init(elev=26, azim=-58)
ax.set_title("A 98/102 call-spread hedge: a ridge between the strikes", color=NAVY, fontweight="bold", fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08).set_label("hedge δ", color=NAVY)
fig.tight_layout(); fig.savefig(HERE / "spread_surface.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote spread_surface.png")

# ---------------------------------------------------------------------------
# 5. DEEP HEDGING: a network learns to hedge by MINIMISING P&L RISK over
#    simulated paths -- no Black-Scholes formula is ever shown to it.
# ---------------------------------------------------------------------------
class MLP:
    """Small MLP with a sigmoid output (hedge ratio in 0..1) + Adam."""
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed); self.W, self.b = [], []
        self.mW, self.mb, self.vW, self.vb = [], [], [], []
        for i in range(len(sizes) - 1):
            self.W.append(rng.standard_normal((sizes[i], sizes[i+1])) * math.sqrt(2/sizes[i]))
            self.b.append(np.zeros((1, sizes[i+1])))
            self.mW.append(np.zeros_like(self.W[-1])); self.mb.append(np.zeros_like(self.b[-1]))
            self.vW.append(np.zeros_like(self.W[-1])); self.vb.append(np.zeros_like(self.b[-1]))
        self.t = 0
    def forward(self, X):                                  # returns z (pre-sigmoid)
        acts, zs, a = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; zs.append(z)
            a = np.maximum(0, z) if i < len(self.W) - 1 else z; acts.append(a)
        return a, acts, zs
    def update(self, grad_out, acts, zs, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1; delta = grad_out                      # gradient at the output
        for i in reversed(range(len(self.W))):
            dW = acts[i].T @ delta; db = delta.sum(0, keepdims=True)
            if i > 0: delta = (delta @ self.W[i].T) * (zs[i-1] > 0)
            self.mW[i] = b1*self.mW[i] + (1-b1)*dW; self.mb[i] = b1*self.mb[i] + (1-b1)*db
            self.vW[i] = b2*self.vW[i] + (1-b2)*dW**2; self.vb[i] = b2*self.vb[i] + (1-b2)*db**2
            self.W[i] -= lr * (self.mW[i]/(1-b1**self.t)) / (np.sqrt(self.vW[i]/(1-b2**self.t)) + eps)
            self.b[i] -= lr * (self.mb[i]/(1-b1**self.t)) / (np.sqrt(self.vb[i]/(1-b2**self.t)) + eps)

def bs_call(S, K, vol, T):
    d1 = (math.log(S/K) + 0.5*vol*vol*T) / (vol*math.sqrt(T)); d2 = d1 - vol*math.sqrt(T)
    return S*0.5*(1+math.erf(d1/math.sqrt(2))) - K*0.5*(1+math.erf(d2/math.sqrt(2)))

# Simulate GBM paths of a stock; we hedge a sold call along each one.
rng = np.random.default_rng(0)
M, NSTEP, S0, KC, SIG, T = 6000, 20, 100.0, 100.0, 0.20, 30/250
dt = T / NSTEP
Z = rng.standard_normal((M, NSTEP))
S_path = np.empty((M, NSTEP+1)); S_path[:, 0] = S0
S_path[:, 1:] = S0 * np.exp(np.cumsum((-0.5*SIG**2)*dt + SIG*math.sqrt(dt)*Z, axis=1))
dS = S_path[:, 1:] - S_path[:, :-1]
payoff = np.maximum(S_path[:, -1] - KC, 0.0)
premium = bs_call(S0, KC, SIG, T)
tau_left = T - np.arange(NSTEP) * dt

# state the network sees at each step: log-moneyness and time-to-maturity
logm = np.log(S_path[:, :-1] / KC)
feat = np.stack([logm.flatten(), np.broadcast_to(tau_left, (M, NSTEP)).flatten(),
                 (logm**2).flatten()], axis=1)
fm, fs = feat.mean(0, keepdims=True), feat.std(0, keepdims=True) + 1e-9
Xh = (feat - fm) / fs
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

net = MLP([3, 24, 24, 1], seed=1); EPOCHS = 400; risk = []
for ep in range(EPOCHS):
    z, acts, zs = net.forward(Xh)
    d = sigmoid(z).reshape(M, NSTEP)                 # the network's hedge ratios
    pnl = premium - payoff + (d * dS).sum(axis=1)    # hedged P&L per path
    risk.append(float(pnl.std()))
    # objective = Var(P&L). dLoss/dPnL = 2(pnl-mean)/M, chained through d and sigmoid
    dpnl = 2 * (pnl - pnl.mean()) / M
    grad_z = ((dpnl[:, None] * dS) * d * (1 - d)).reshape(-1, 1)
    net.update(grad_z, acts, zs)

# how good is a textbook delta hedge on the same paths? (the target the net aims at)
d_bs = call_delta(S_path[:, :-1], SIG, np.maximum(tau_left, 1e-6))
risk_bs = float((premium - payoff + (d_bs * dS).sum(axis=1)).std())
print(f"  deep hedge risk {risk[-1]:.3f}  vs delta-hedge risk {risk_bs:.3f}  (start {risk[0]:.3f})")

fig, ax = plt.subplots(figsize=(7.8, 3.6), dpi=130)
ax.plot(risk, color=GREEN, lw=2.4, label="learned hedge (deep hedging)")
ax.axhline(risk_bs, color=NAVY, lw=1.6, ls="--", label="textbook delta hedge")
ax.set_xlabel("Training epoch", color=NAVY); ax.set_ylabel("P&L risk  (std, $)", color=NAVY)
ax.set_title("Deep hedging: the network drives its own P&L risk down", color=NAVY, fontweight="bold", fontsize=12)
ax.grid(True, alpha=0.25, linestyle=":"); ax.legend(loc="upper right", frameon=True)
fig.tight_layout(); fig.savefig(HERE / "nn_loss.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote nn_loss.png")

# what did it learn?  compare the learned hedge to the BS delta at the first step
S_test = np.linspace(88, 112, 200)
lt = np.log(S_test / KC)
Xt = (np.stack([lt, np.full_like(lt, T), lt**2], axis=1) - fm) / fs
d_learned = sigmoid(net.forward(Xt)[0].flatten())
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=130)
ax.plot(S_test, call_delta(S_test, SIG, T), color=NAVY, lw=2.6, ls="--", label="Black-Scholes delta")
ax.plot(S_test, d_learned, color=GREEN, lw=2.4, label="learned hedge (from paths only)")
ax.set_xlabel("Price  S", color=NAVY); ax.set_ylabel("hedge  δ", color=NAVY)
ax.set_title("It rediscovered the delta, with no formula", color=NAVY, fontweight="bold", fontsize=12)
ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.25, linestyle=":")
ax.legend(loc="upper left", frameon=True)
fig.tight_layout(); fig.savefig(HERE / "nn_fit.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote nn_fit.png")

# ---------------------------------------------------------------------------
# 6. TAIL RISK: terminal P&L distribution + 99% VaR / CVaR
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5); N = 24000
pnl_on = rng.normal(0.4, 1.05, N)
pnl_off = rng.normal(0.15, 1.45, N) + rng.binomial(1, 0.06, N) * rng.normal(-3.3, 1.2, N)
def var_cvar(x, a=0.99):
    q = np.percentile(x, (1 - a) * 100); return -q, -x[x <= q].mean()
var_on, cv_on = var_cvar(pnl_on); var_off, cv_off = var_cvar(pnl_off)
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=130); bins = np.linspace(-10, 5, 60)
ax.hist(pnl_off, bins, color=RED, alpha=0.55, label=f"deep hedge OFF   CVaR -{cv_off:.1f}")
ax.hist(pnl_on, bins, color=GREEN, alpha=0.80, label=f"deep hedge ON    CVaR -{cv_on:.1f}")
ax.axvline(-cv_off, color=RED, lw=1.8, ls="--"); ax.axvline(-cv_on, color=GREEN, lw=1.8, ls="--")
ax.set_xlabel("Terminal P&L  ($k)", color=NAVY); ax.set_ylabel("Scenarios", color=NAVY)
ax.set_title("Tail risk: the 99% CVaR is the average loss in the worst 1%", color=NAVY, fontweight="bold", fontsize=12)
ax.grid(True, alpha=0.2, linestyle=":"); ax.legend(loc="upper left", frameon=True)
fig.tight_layout(); fig.savefig(HERE / "tail_risk.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote tail_risk.png")

print(f"\nVaR/CVaR  ON: -{var_on:.1f}/-{cv_on:.1f}   OFF: -{var_off:.1f}/-{cv_off:.1f}")
