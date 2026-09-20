"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install gs-quant yfinance pandas
import pandas as pd, yfinance as yf
import gs_quant.timeseries as ts
from gs_quant.timeseries import Window

# ---------- data collection: free, no account ----------
px = yf.download(["SPY", "TLT"], start="2010-01-01", end="2024-01-01",
                 progress=False, auto_adjust=True)["Close"]
px = px.dropna(axis=1, thresh=250).ffill(limit=1).dropna(how="any")

out = pd.DataFrame(index=px.index)
out["vol_63"]  = ts.volatility(px.SPY, "3m")
out["corr_63"] = ts.correlation(px.SPY, px.TLT, Window(63, 0))
out["beta_63"] = ts.beta(px.SPY, px.TLT, Window(63, 0))
