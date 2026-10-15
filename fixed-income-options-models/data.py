"""Free data, cached under data/: FRED, Yahoo Finance and SEC company facts.

Every series is cut at AS_OF, so a run on a later day reproduces the numbers in the README.
Pass --refresh to download everything again, or set AS_OF to move the date.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

AS_OF = pd.Timestamp(os.environ.get("AS_OF", "2026-10-13"))
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
REFRESH = "--refresh" in sys.argv


def fred(series: str, start: str = "1990-01-01") -> pd.Series:
    """One FRED series as a float Series, from `start` to AS_OF."""
    f = DATA / f"fred_{series}.csv"
    if f.exists() and not REFRESH:
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].dropna()
        if s.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=31):
            return s.loc[:AS_OF]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    # curl rather than urllib: FRED resets some Python connections but serves curl's client
    for attempt in range(4):
        raw = subprocess.run(["curl", "-s", "-m", "90", url], capture_output=True, text=True).stdout
        if raw.count("\n") > 2:
            break
        time.sleep(3 + 3 * attempt)
    else:
        raise RuntimeError(f"FRED did not answer for {series}")
    df = pd.read_csv(io.StringIO(raw))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=series).dropna()
    s.to_frame().to_csv(f)
    return s.loc[:AS_OF]


def prices(ticker: str, start: str, adjusted: bool) -> pd.Series:
    """Daily closes from Yahoo Finance. Unadjusted for market caps, adjusted for returns."""
    f = DATA / f"yahoo_{ticker}_{'adj' if adjusted else 'raw'}.csv"
    if not f.exists() or REFRESH:
        import yfinance as yf
        s = yf.download(ticker, start=start, auto_adjust=adjusted, progress=False)["Close"]
        s = (s[ticker] if isinstance(s, pd.DataFrame) else s).dropna().rename(ticker)
        s.to_frame().to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].dropna().loc[:AS_OF]


def sec_facts(ticker: str, cik: int) -> dict:
    """SEC company facts (XBRL). The SEC asks for a User-Agent with a name and an email:
    set SEC_USER_AGENT, e.g. "Jane Doe jane@example.com"."""
    f = DATA / f"sec_{ticker}.json"
    if f.exists() and not REFRESH:
        return json.loads(f.read_text())
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT to 'Your Name your@email' for the SEC API.")
    req = urllib.request.Request(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                                 headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    f.write_bytes(raw)
    return json.loads(raw)


def _tag(gaap: dict, tag: str) -> pd.Series:
    """Balance sheet value of one XBRL tag by period end ($bn), from 10-Q and 10-K filed
    by AS_OF; when a period was filed twice, the later filing wins."""
    if tag not in gaap:
        return pd.Series(dtype=float)
    v = [x for x in gaap[tag]["units"].get("USD", [])
         if x.get("form") in ("10-Q", "10-K") and "start" not in x and pd.Timestamp(x["filed"]) <= AS_OF]
    if not v:
        return pd.Series(dtype=float)
    df = pd.DataFrame(v).sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
    return pd.Series(df["val"].values / 1e9, index=pd.to_datetime(df["end"]))


def debt_and_shares(ticker: str, cik: int) -> tuple[pd.DataFrame, pd.Series]:
    """Short and long debt ($bn) by quarter end, and shares outstanding (bn).
    Short: DebtCurrent, or the current part of long term debt plus commercial paper.
    Long: the noncurrent line, under whichever of its two common tags is more recent."""
    j = sec_facts(ticker, cik)
    g = j["facts"]["us-gaap"]
    dc = _tag(g, "DebtCurrent")
    short = dc if len(dc) else _tag(g, "LongTermDebtCurrent").add(_tag(g, "CommercialPaper"), fill_value=0)
    long_ = _tag(g, "LongTermDebtNoncurrent")
    alt = _tag(g, "LongTermDebtAndCapitalLeaseObligations")
    if len(alt) and (not len(long_) or alt.index[-1] > long_.index[-1]):
        long_ = alt
    debt = pd.concat({"short": short, "long": long_}, axis=1).sort_index().ffill().dropna()
    sh = pd.DataFrame(j["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"])
    sh = sh[pd.to_datetime(sh["filed"]) <= AS_OF].sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
    shares = pd.Series(sh["val"].values / 1e9, index=pd.to_datetime(sh["end"]))
    return debt, shares
