"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install kagglehub arch yfinance pandas numpy pyarrow
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

# ---------- data collection: Kaggle as an API call, not a manual download ----------
# needs a free Kaggle token, either ~/.kaggle/kaggle.json or the env vars
# KAGGLE_USERNAME / KAGGLE_KEY. 14 parquet files, about 600 MB, cached after the
# first call so re-running costs nothing.
import kagglehub
KAGGLE = Path(kagglehub.dataset_download(
    "dudesurfin/spy-options-eod-volatility-surface-2010-2023"))
# -> spy_eod_2010.parquet ... spy_eod_2023.parquet


def atm_implied(years=range(2016, 2024), dte=(25, 35)):
    """30 day at the money implied vol, one point per day, from the Kaggle SPY chains.
    Averaging the call and put IV at the nearest strike cancels most of the
    put call parity noise you get in a free end of day snapshot."""
    out = []
    for y in years:
        f = KAGGLE / f"spy_eod_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d.columns = [c.strip().strip("[]") for c in d.columns]
        for c in ("UNDERLYING_LAST", "STRIKE", "DTE", "C_IV", "P_IV"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["QUOTE_DATE"] = pd.to_datetime(d["QUOTE_DATE"])
        d = d[d.DTE.between(*dte)].copy()
        d["dist"] = (d.STRIKE - d.UNDERLYING_LAST).abs()
        atm = d.loc[d.groupby("QUOTE_DATE").dist.idxmin()]        # nearest strike each day
        out.append(atm.set_index("QUOTE_DATE")[["C_IV", "P_IV"]].mean(axis=1) * 100)
    iv = pd.concat(out).sort_index()
    return iv[(iv > 3) & (iv 0).mean():.1f}% of days")
print(f"worst day      {j.vrp.min():.1f} on {j.vrp.idxmin().date()}")
