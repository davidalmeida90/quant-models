"""GJR-GARCH(1,1) with Student t errors on SPY daily returns since 2005.

Fits the model, prints persistence, half life and long run volatility, builds the forecast
curve for every horizon up to a year, then runs an out of sample test from 2020 with
parameters estimated on data to 2019 only, against realised vol and the VIX.

    py -3 gjr_garch.py

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
import numpy as np, pandas as pd, yfinance as yf
from arch import arch_model

spy = yf.download("SPY", start="2005-01-01", auto_adjust=True)["Close"].squeeze()
r = 100 * np.log(spy).diff().dropna()

# o=1 adds the GJR term: gamma fires only on down days. Student t for the fat tails.
res = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
a, g, b = res.params["alpha[1]"], res.params["gamma[1]"], res.params["beta[1]"]
lam = a + b + g / 2
print(f"alpha {a:.3f}, gamma {g:.3f}, beta {b:.3f}, persistence {lam:.3f}")
print(f"half life {np.log(0.5) / np.log(lam):.0f} days, "
      f"long run {np.sqrt(252 * res.params['omega'] / (1 - lam)):.1f}%")

# average vol expected over every horizon up to a year, to set against the VIX curve
var = res.forecast(horizon=252, reindex=False).variance.values[-1]
curve = np.sqrt(252 * np.cumsum(var) / np.arange(1, 253))


# out of sample: parameters estimated ONLY on data to 2019, then filtered forward
p19 = arch_model(r[:"2019-12-31"], p=1, o=1, q=1, dist="t").fit(disp="off").params
s2 = arch_model(r, p=1, o=1, q=1, dist="t").fix(p19).conditional_volatility.values ** 2
lam19 = p19["alpha[1]"] + p19["beta[1]"] + p19["gamma[1]"] / 2
long_run = p19["omega"] / (1 - lam19)
eps = r.values - p19["mu"]
s2_next = p19["omega"] + (p19["alpha[1]"] + p19["gamma[1]"] * (eps < 0)) * eps ** 2 + p19["beta[1]"] * s2
decay = (lam19 ** np.arange(21)).mean()                # closed form, 21 day average
forecast = np.sqrt(252 * (long_run + decay * (s2_next - long_run)))
realised = np.sqrt(252 * pd.Series(r.values ** 2).rolling(21).mean().shift(-21).values)

vix = yf.download("^VIX", start="2005-01-01")["Close"].squeeze().reindex(r.index).values
d = pd.DataFrame(dict(garch=forecast, vix=vix, real=realised), index=r.index)["2020":].dropna()
print(d.corr()["real"].round(2))
