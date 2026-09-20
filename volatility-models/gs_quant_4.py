"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install gs-quant yfinance pandas numpy
import pandas as pd, yfinance as yf
import gs_quant.timeseries as ts
from gs_quant.timeseries import volatility, percentiles, last_value, Window

# ---------- data collection: the nine G10 pairs, free, no account ----------
PAIRS = {c: (c[3:] + "=X" if c.startswith("USD") else c + "=X") for c in
         ['USDJPY','EURUSD','AUDUSD','GBPUSD','USDCAD','USDNOK','NZDUSD','USDSEK','USDCHF']}
raw = yf.download(list(PAIRS.values()), start="2018-01-01", end="2024-01-01",
                  progress=False, auto_adjust=True)["Close"]
raw.columns = [{v: k for k, v in PAIRS.items()}[c] for c in raw.columns]

n = raw["USDNOK"].dropna()
pd.DataFrame({
    "raw":             volatility(n, "3m"),
    "smooth_spikes":   volatility(ts.smooth_spikes(n, 0.10), "3m"),
    "smooth_outliers": volatility(ts.smooth_outliers(n, 0.5), "3m"),
    "winsorize":       volatility(ts.prices(ts.winsorize(ts.returns(n), 2.5, Window(252, 0))), "3m"),
}).describe()
