"""Market data for the backtest reels. Downloaded once, cached to disk.

Everything below is real. Adjusted daily closes from Yahoo, pulled once and
parked in backtest_videos/_data/ so a re-render never touches the network.

Two universes:

    ETFS    45 liquid ETFs spanning the four asset classes Moskowitz, Ooi and
            Pedersen use for time series momentum: equity indices, bonds,
            commodities and currencies.
    STOCKS  ~100 large cap US names, used for cross sectional momentum and for
            the pairs.

One honest warning that belongs on screen wherever STOCKS is used: this list is
today's large caps, so it is a survivor list. Anything that went to zero or got
taken out is missing, and that inflates the LEVEL of every return in a cross
sectional test. Shape of the parameter surface is what these reels are about,
and shape survives the bias. Level does not, so it never gets quoted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "_data"
START = "2005-01-01"

ETFS = [
    # equity indices
    "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "EWJ", "EWG", "EWU",
    "EWZ", "EWY", "EWA", "EWC", "FXI", "EWH", "EWW",
    # bonds
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "BWX", "EMB",
    # commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "DBO", "PPLT",
    # currencies
    "FXE", "FXY", "FXB", "FXA", "FXF", "FXC", "UUP",
    # real assets
    "IYR", "VNQ", "XLE",
]

STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "PG", "HD", "LOW", "XOM", "CVX", "COP", "SLB", "PFE",
    "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS",
    "KO", "PEP", "MCD", "SBUX", "NKE", "WMT", "TGT", "COST", "CL", "KMB",
    "MMM", "CAT", "DE", "HON", "GE", "BA", "LMT", "NOC", "RTX", "UPS",
    "UNP", "CSX", "NSC", "FDX", "T", "VZ", "CMCSA", "DIS", "NFLX", "ORCL",
    "CRM", "ADBE", "INTC", "AMD", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX",
    "IBM", "CSCO", "ACN", "NOW", "INTU", "BAC", "WFC", "C", "GS", "MS",
    "BLK", "SCHW", "AXP", "SPGI", "CB", "PGR", "TRV", "AIG", "MET", "PRU",
    "DUK", "SO", "NEE", "D", "AEP", "EXC", "SRE", "PSA", "SPG", "O",
]

MACRO = ["SPY", "^VIX"]


def _pull(tickers, tag):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf
    print(f"[data] downloading {len(tickers)} tickers for {tag} ...")
    raw = yf.download(tickers, start=START, progress=False, auto_adjust=True,
                      threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(how="all").ffill(limit=3)
    px.to_parquet(path)
    print(f"[data] {tag}: {px.shape[0]} days x {px.shape[1]} names -> {path.name}")
    return px


def etfs(min_days=3200):
    px = _pull(ETFS, "etfs")
    keep = [c for c in px.columns if px[c].notna().sum() >= min_days]
    return px[keep].dropna(how="all")


def stocks(min_days=4000):
    px = _pull(STOCKS, "stocks")
    keep = [c for c in px.columns if px[c].notna().sum() >= min_days]
    return px[keep].dropna(how="all")


def spy_vix():
    px = _pull(MACRO, "macro")
    px = px.dropna()
    return px["SPY"].to_numpy(float), px["^VIX"].to_numpy(float) / 100.0, px.index


def logret(px: pd.DataFrame) -> np.ndarray:
    """Simple returns, NaN where the name has no history yet."""
    return px.pct_change().to_numpy(float)


# Which asset class each ETF belongs to, so a correlation matrix can be ordered
# into blocks instead of alphabetically.
GROUPS = [
    ("EQUITY", ["SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "EWJ", "EWG",
                "EWU", "EWZ", "EWY", "EWA", "EWC", "FXI", "EWH", "EWW"]),
    ("BONDS", ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "BWX", "EMB"]),
    ("COMMOD", ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "DBO", "PPLT"]),
    ("FX", ["FXE", "FXY", "FXB", "FXA", "FXF", "FXC", "UUP"]),
    ("REAL", ["IYR", "VNQ", "XLE"]),
]


def group_order(names):
    """Return (indices in block order, [(label, start, stop), ...])."""
    idx, blocks = [], []
    for label, members in GROUPS:
        start = len(idx)
        for t in members:
            if t in names:
                idx.append(list(names).index(t))
        if len(idx) > start:
            blocks.append((label, start, len(idx)))
    return idx, blocks


if __name__ == "__main__":
    e = etfs()
    s = stocks()
    sp, vx, idx = spy_vix()
    print(f"etfs   {e.shape[0]} days x {e.shape[1]} names  "
          f"{e.index[0].date()} -> {e.index[-1].date()}")
    print(f"stocks {s.shape[0]} days x {s.shape[1]} names  "
          f"{s.index[0].date()} -> {s.index[-1].date()}")
    print(f"macro  {len(sp)} days  {idx[0].date()} -> {idx[-1].date()}  "
          f"VIX mean {vx.mean()*100:.1f}")
