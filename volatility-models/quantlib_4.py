"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install QuantLib kagglehub pandas numpy pyarrow
from pathlib import Path
import numpy as np, pandas as pd, QuantLib as ql

# ---------- collection: Kaggle as an API call ----------
import kagglehub
KAGGLE = Path(kagglehub.dataset_download(
    "dudesurfin/spy-options-eod-volatility-surface-2010-2023"))

d = pd.read_parquet(KAGGLE / "spy_eod_2023.parquet")
d.columns = [c.strip().strip("[]") for c in d.columns]
for c in ("UNDERLYING_LAST", "STRIKE", "DTE", "C_IV", "P_IV"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[pd.to_datetime(d.QUOTE_DATE) == "2023-06-15"]
spot = float(d.UNDERLYING_LAST.iloc[0])

# ---------- curation, four filters that all change the fit ----------
rows = []
for _, x in d.iterrows():
    if not (25  0.12:                   # the far wings have the widest spreads and the
        continue                        # least reliable mids, and they dominate an
                                        # unweighted least squares if you let them
    iv = x.P_IV if x.STRIKE < spot else x.C_IV   # OTM side only, always the liquid one
    if not (0.05 < iv < 1.0):
        continue
    rows.append((int(x.DTE), float(x.STRIKE), float(iv)))

q = pd.DataFrame(rows, columns=["dte", "strike", "iv"]).dropna()
# thin to at most 9 strikes per expiry, so no single dense expiry outvotes the others
q = pd.concat([g.iloc[np.linspace(0, len(g)-1, min(9, len(g))).astype(int)]
               for _, g in q.groupby("dte")], ignore_index=True)

# ---------- QuantLib ----------
today = ql.Date(15, 6, 2023); ql.Settings.instance().evaluationDate = today
cal, dc = ql.TARGET(), ql.Actual365Fixed()
rf = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.050, dc))
dv = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.015, dc))
sh = ql.QuoteHandle(ql.SimpleQuote(spot))

proc = ql.HestonProcess(rf, dv, sh, 0.030, 3.0, 0.030, 0.6, -0.75)   # v0,kappa,theta,sigma,rho
model = ql.HestonModel(proc)
engine = ql.AnalyticHestonEngine(model)

helpers = []
for _, x in q.iterrows():
    h = ql.HestonModelHelper(ql.Period(int(x.dte), ql.Days), cal, spot, float(x.strike),
                             ql.QuoteHandle(ql.SimpleQuote(float(x.iv))), rf, dv)
    h.setPricingEngine(engine)
    helpers.append(h)

# THE SAME TRAP AGAIN, and this time it fails SILENTLY. The constraint vector is in
# params() order, NOT constructor order. Write the bounds in constructor order and the
# starting point lands outside the feasible region, LevenbergMarquardt returns without
# taking a single step, and model.params() hands back your initial guess looking like
# a converged answer. Simplex at least raises "Initial guess is not in the feasible
# region". LM says nothing at all.
LO = [1e-4, 0.5, 1e-4, -0.999, 1e-4]      # theta, kappa, sigma, rho, v0
HI = [1.00, 15.0, 5.00,  0.000, 1.000]

model.calibrate(helpers, ql.LevenbergMarquardt(),
                ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8),
                ql.NonhomogeneousBoundaryConstraint(LO, HI))

theta, kappa, sigma, rho, v0 = model.params()      # <-- NOT constructor order
print(f"v0={v0:.5f} kappa={kappa:.4f} theta={theta:.5f} sigma={sigma:.4f} rho={rho:.4f}")
print(f"  spot vol      {100*np.sqrt(v0):.2f}%")
print(f"  long run vol  {100*np.sqrt(theta):.2f}%")
print(f"  Feller 2*k*theta={2*kappa*theta:.4f} vs sigma^2={sigma**2:.4f}")
print(f"  mean relative pricing error {100*np.mean([abs(h.calibrationError()) for h in helpers]):.2f}%")
