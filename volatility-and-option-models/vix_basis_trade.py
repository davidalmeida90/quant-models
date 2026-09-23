"""VIX futures basis trade (Simon and Campasano, 2014) on the Cboe short VIX futures index.

Short volatility only while VIX closes below VIX3M, decided at yesterday's close, 10 basis
points per switch, against always short and against the long in backwardation version.

    py -3 vix_basis_trade.py

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
import numpy as np, pandas as pd, yfinance as yf

px = yf.download(["^VIX", "^VIX3M", "^SHORTVOL"], start="2006-07-17")["Close"].dropna()
ret = px["^SHORTVOL"].pct_change()                     # Cboe short VIX futures index

# decided at yesterday's close, applied to today's return
short = (px["^VIX"] < px["^VIX3M"]).astype(float).shift(1)
switch = short.diff().abs().fillna(0)
rules = pd.DataFrame({
    "always short": ret,
    "short in contango": short * ret - 0.001 * switch,
    "short contango, long backwardation": (2 * short - 1) * ret - 0.002 * switch,
}).dropna()

def stats(x):
    g = (1 + x).cumprod(); years = len(x) / 252
    return pd.Series({"CAGR %": 100 * (g.iloc[-1] ** (1 / years) - 1),
                      "vol %": 100 * x.std() * np.sqrt(252),
                      "Sharpe": x.mean() / x.std() * np.sqrt(252),
                      "max drawdown %": 100 * (g / g.cummax() - 1).min(),
                      "worst day %": 100 * x.min()})

print(rules.apply(stats).T.round(2))
print(f"contango on {(px['^VIX'] < px['^VIX3M']).mean():.0%} of days")
