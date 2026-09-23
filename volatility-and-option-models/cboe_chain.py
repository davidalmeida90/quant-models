"""SPY option chain from the Cboe delayed quotes feed, with the forward and Black 76 helpers.

Imported by straddle_calendar.py and heston_calibration.py. Downloads the chain once, at
import, with no API key.

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
import numpy as np, pandas as pd, requests
from scipy.stats import norm

# Cboe delayed quotes: every SPY option with bid, ask and last trade, no key needed
url = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
raw = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()["data"]
spot = float(raw["close"])

rows = []
for o in raw["options"]:
    s = o["option"]                                    # e.g. SPY261023C00775000
    rows.append(dict(expiry=pd.Timestamp("20" + s[3:9]), kind=s[9],
                     strike=int(s[10:]) / 1000, bid=o["bid"], ask=o["ask"],
                     last=o.get("last_trade_time")))
chain = pd.DataFrame(rows)
chain["mid"] = (chain.bid + chain.ask) / 2
asof = pd.to_datetime(chain["last"].dropna()).max().normalize()
chain["T"] = (chain.expiry - asof).dt.days / 365
r = 0.040                                              # 13 week T-bill, ^IRX

def forward(expiry):
    """SPY options are American, so parity only gives the forward, never the discount."""
    g = chain[chain.expiry == expiry]
    T = g["T"].iloc[0]; D = np.exp(-r * T)
    c, p = (g[g.kind == k].set_index("strike") for k in ("C", "P"))
    ks = c.index.intersection(p.index)
    ks = ks[abs(ks / spot - 1) < 0.02]
    return float(np.median(ks.values + (c.loc[ks, "mid"].values - p.loc[ks, "mid"].values) / D)), D, T

def black(F, K, T, D, vol, call=True):
    d1 = (np.log(F / K) + vol * vol * T / 2) / (vol * np.sqrt(T)); d2 = d1 - vol * np.sqrt(T)
    if call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol(price, F, K, T, D, call=True):
    lo, hi = 1e-4, 3.0
    for _ in range(60):                                # bisection, error below 1e-15
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if black(F, K, T, D, mid, call) > price else (mid, hi)
    return (lo + hi) / 2
