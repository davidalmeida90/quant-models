"""gs quant, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install scipy pandas numpy
import numpy as np, pandas as pd
from scipy.optimize import least_squares
rng = np.random.default_rng(7)

# ---------- data collection: one live chain from Cboe, no account, no key ----------
import json, urllib.request
url = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
raw = json.loads(urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read())
spot = float(raw["data"]["current_price"])

import re
OCC = re.compile(r"^([A-Z^_]+?)(\d{6})([CP])(\d{8})$")
rows = []
for c in raw["data"]["options"]:
    m = OCC.match(c["option"])
    if not m or not c["iv"] or c["bid"] = spot) & (d.side == "C")) | ((d.strike < spot) & (d.side == "P"))]  # OTM side

g = d[d.dte == 46]
T = 46 / 365
k_all, w_all = np.log(g.strike / spot).values, (g.iv.values ** 2) * T


def svi(p, k):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit(k, w):
    r = least_squares(lambda p: svi(p, k) - w, [w.mean(), .1, -.5, 0., .1],
                      bounds=([-np.inf, 0, -.999, -np.inf, 1e-6],
                              [np.inf, np.inf, .999, np.inf, np.inf]), max_nfev=20000)
    return r.x


for n in (10, 16, 24, 40, 80, 160, len(k_all)):
    rhos = [fit(k_all[i], w_all[i])[2]
            for i in (rng.choice(len(k_all), n, replace=False) for _ in range(300))]
    print(f"n={n:4d}  rho median {np.median(rhos):+.4f}  "
          f"5-95 {np.percentile(rhos,5):+.3f} to {np.percentile(rhos,95):+.3f}")
