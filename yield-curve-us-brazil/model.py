"""US and Brazil yield curves, from the write-up at
https://davidariasfinance.com/scripts/yield-curve-us-brazil/

FRED for the US constant maturity series, BCB SGS 432 and Tesouro Direto for Brazil,
with a Nelson Siegel Svensson fit where the Brazilian data is sparse.
"""

from fredapi import Fred
import pandas as pd, numpy as np

MATS   = [1, 3, 6, 12, 24, 36, 60, 84, 120]          # months
SERIES = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
          "DGS3", "DGS5", "DGS7", "DGS10"]
START  = "2016-01-01"

fred = Fred(api_key=key)
df = pd.DataFrame({m: fred.get_series(s, observation_start=START)
                   for m, s in zip(MATS, SERIES)})
df["VIX"] = fred.get_series("VIXCLS", observation_start=START)

# Month end last print. interpolate covers holidays, it does not invent regimes.
df = (df.resample("ME").last()
        .interpolate(limit_direction="both")
        .dropna(how="all"))

# 10y minus 3m, the spread the NY Fed uses for its recession model
slope = df[120] - df[3]

# ---------------------------------------------------------------------------

def ns(tau, b0, b1, b2, lam=1.8):
    """Nelson Siegel. Three betas, one decay.

    b0 is the level, the rate a very long bond converges to.
    b1 is the slope, and it loads hardest on the short end.
    b2 is the curvature, and it loads on the belly around lam.
    """
    x  = tau / lam
    g1 = np.where(x > 1e-9, (1 - np.exp(-x)) / x, 1.0)   # slope loading
    g2 = g1 - np.exp(-x)                                    # curvature loading
    return b0 + b1 * g1 + b2 * g2

# ---------------------------------------------------------------------------

# SGS 432 = Selic target. Daily series is capped at 10y per query, so chunk it.
UA  = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
       "Referer": "https://www3.bcb.gov.br/"}
url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"

chunks = [("01/01/2015", "31/12/2019"),
          ("01/01/2020", date.today().strftime("%d/%m/%Y"))]

parts = []
for di, dfim in chunks:
    for _ in range(4):                       # SGS drops requests under load
        r = requests.get(url, params={"formato": "json",
                         "dataInicial": di, "dataFinal": dfim},
                         headers=UA, timeout=60)
        if r.status_code == 200 and r.text.strip().startswith("["):
            d = pd.DataFrame(r.json())
            d["data"]  = pd.to_datetime(d["data"], dayfirst=True)
            d["valor"] = d["valor"].astype(float)
            parts.append(d.set_index("data")["valor"]); break
        time.sleep(3)

# ---------------------------------------------------------------------------

td = pd.read_csv(TESOURO, sep=";", decimal=",")          # BR conventions
td["Data Base"]       = pd.to_datetime(td["Data Base"], dayfirst=True)
td["Data Vencimento"] = pd.to_datetime(td["Data Vencimento"], dayfirst=True)

pref = td[td["Tipo Titulo"].isin(["Tesouro Prefixado",
          "Tesouro Prefixado com Juros Semestrais"])].copy()

# Time to maturity in years. This is the number FRED hides from you.
pref["ttm"]   = (pref["Data Vencimento"] - pref["Data Base"]).dt.days / 365.25
pref["yield"] = pref["Taxa Compra Manha"]
pref = pref[pref["ttm"] > 0.05]        # about to mature, price is noise

# One observation per month: the last trading day.
pref["ym"] = pref["Data Base"].dt.to_period("M")
last = pref.groupby("ym")["Data Base"].max()
pref = pref[pref["Data Base"].isin(last.values)]

rows = {}
for d, g in pref.groupby("Data Base"):
    pts = g[["ttm", "yield"]].drop_duplicates("ttm").sort_values("ttm").values
    if len(pts) < 4:                  # fewer than 4 points cannot pin 6 params
        continue
    st = selic.loc[selic.index <= d].iloc[-1]    # anchor the short end
    p  = fit_nss(pts[:, 0], pts[:, 1], st)
    rows[d] = nss(np.array(MATS) / 12.0, *p)      # read at fixed tenors
