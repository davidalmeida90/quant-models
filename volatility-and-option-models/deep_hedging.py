"""Deep hedging (Buehler, Gonon, Teichmann and Wood, 2019) in PyTorch.

A network hedges a short 30 day at the money call once a day, paying 10 basis points on every
dollar traded, and is trained end to end on CVaR 95% of the loss. It sees moneyness, time
left and the shares it already holds. Tested on 500,000 fresh paths against Black-Scholes
delta. Runs on a GPU when there is one; 3,000 training steps take a few minutes there.

    py -3 deep_hedging.py

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
import math, numpy as np, torch, torch.nn as nn
DEV = "cuda" if torch.cuda.is_available() else "cpu"

S0 = K = 100.0; SIG = 0.20; N = 30; T = N / 252; DT = T / N
C = 0.001                                               # 10 bp on every dollar traded
d1 = (SIG * SIG * T / 2) / (SIG * math.sqrt(T))
PREMIUM = S0 * 0.5 * (1 + math.erf(d1 / 2 ** 0.5)) - K * 0.5 * (1 + math.erf((d1 - SIG * math.sqrt(T)) / 2 ** 0.5))
TAU = torch.tensor([T - k * DT for k in range(N)], device=DEV)

def paths(n):
    z = torch.randn(n, N, device=DEV)
    ls = math.log(S0) + torch.cumsum(-SIG ** 2 / 2 * DT + SIG * math.sqrt(DT) * z, 1)
    return torch.cat([torch.full((n, 1), S0, device=DEV), torch.exp(ls)], 1)

net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                    nn.Linear(64, 1)).to(DEV)

def policy(t, S_t, held):
    m = torch.log(S_t / K) / (SIG * torch.sqrt(TAU[t]))
    x = torch.stack([m.clamp(-4, 4), (TAU[t] / T).expand_as(m), held], 1)
    return torch.sigmoid(net(x)).squeeze(1)             # a call delta lives in [0, 1]

def hedge_pnl(S, rule, c=C):
    """Seller of the call: premium in, hedge gains, trading costs, payoff out."""
    held = torch.zeros(S.shape[0], device=DEV)
    pnl = torch.full_like(held, PREMIUM)
    for t in range(N):
        new = rule(t, S[:, t], held)
        pnl += new * (S[:, t + 1] - S[:, t]) - c * S[:, t] * (new - held).abs()
        held = new
    return pnl - c * S[:, -1] * held - torch.clamp(S[:, -1] - K, min=0)

w = torch.zeros(1, device=DEV, requires_grad=True)     # the VaR, learned jointly
opt = torch.optim.Adam(list(net.parameters()) + [w], lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
for step in range(3000):
    pnl = hedge_pnl(paths(16384), policy)
    loss = w + torch.relu(-pnl - w).mean() / 0.05       # CVaR 95%, Rockafellar and Uryasev
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def bs_delta(t, S_t, held):
    d = (torch.log(S_t / K) + SIG ** 2 / 2 * TAU[t]) / (SIG * torch.sqrt(TAU[t]))
    return 0.5 * (1 + torch.erf(d / 2 ** 0.5))

def cvar(loss, a=0.95):
    return loss[loss >= np.quantile(loss, a)].mean()

with torch.no_grad():
    S = paths(500_000)                                  # never seen in training
    net_pnl = hedge_pnl(S, policy).cpu().numpy()
    bs_pnl = hedge_pnl(S, bs_delta).cpu().numpy()
print(f"CVaR 95%: network {cvar(-net_pnl):.3f}, delta {cvar(-bs_pnl):.3f}")
