"""tf quant finance, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install kagglehub scipy pandas numpy pyarrow
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

# ---------- data collection: Kaggle as an API call ----------
import kagglehub
KAGGLE = Path(kagglehub.dataset_download(
    "dudesurfin/spy-options-eod-volatility-surface-2010-2023"))


def load_chain(date="2023-06-15", dte=(25, 45)):
    """One expiry from the historical file, out of the money side only.

    Out of the money on both sides is deliberate. In the money quotes are mostly
    intrinsic value, so their implied vol is numerically unstable and will drag a fit."""
    d = pd.read_parquet(KAGGLE / f"spy_eod_{date[:4]}.parquet")
    d.columns = [c.strip().strip("[]") for c in d.columns]
    for c in ("UNDERLYING_LAST", "STRIKE", "DTE", "C_IV", "P_IV"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(pd.to_datetime(d.QUOTE_DATE) == date) & d.DTE.between(*dte)]

    F = float(d.UNDERLYING_LAST.iloc[0])           # forward proxy, spot is close enough here
    T = float(d.DTE.iloc[0]) / 365.0
    puts = d[(d.STRIKE = F) & d.C_IV.between(.03, 2)][["STRIKE", "C_IV"]].rename(columns={"C_IV": "iv"})
    q = pd.concat([puts, calls]).dropna()
    q["k"] = np.log(q.STRIKE / F)                  # log moneyness
    q["w"] = q.iv ** 2 * T                         # total implied variance
    return q[q.k.between(-0.25, 0.15)], F, T       # trim the far wings, they are noise


def svi(k, p):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(q):
    k, w = q.k.values, q.w.values

    def loss(p):
        # the constraints are not decoration: b=1 gives arbitrage
        if p[1] = 0.999 or p[4] <= 1e-4:
            return 1e9
        return np.mean((svi(k, p) - w) ** 2)

    best = minimize(loss, [w.mean(), 0.1, -0.7, 0.0, 0.1], method="Nelder-Mead",
                    options={"maxiter": 40000, "xatol": 1e-12, "fatol": 1e-14})
    return best.x


def vol_at(p, k, T):
    """Implied vol at any log moneyness, including strikes nobody quoted."""
    return np.sqrt(svi(np.asarray(k), p) / T)


q, F, T = load_chain("2023-06-15")
p = fit_svi(q)
rmse = np.sqrt(np.mean((vol_at(p, q.k.values, T) - q.iv.values) ** 2))

print(f"{len(q)} quotes, forward {F:.2f}, T {T:.3f}")
print(f"fit RMSE {100*rmse:.3f} vol points")
for k, label in [(-0.10, "10% OTM put"), (0.0, "at the money"), (0.05, "5% OTM call")]:
    print(f"  {label:14s} {100*vol_at(p, k, T):.2f}%")
