"""Second and third order option greeks on a real option chain.

Backs the YouTube script HIGHER_ORDER_GREEKS_SCRIPT.md. Every analytic formula in
here is checked against a finite difference of the Black-Scholes-Merton price
surface before any number is used on screen, so a sign error cannot reach the video.

Conventions, stated once because published tables disagree:
  tau   time to maturity in YEARS
  t     calendar time, so d/dt = -d/dtau
  q     continuous dividend yield
  Time derivatives (theta, charm, veta, color) are reported PER CALENDAR DAY as
  d/dt divided by 365, which is why they are negative for a long option.
  sigma derivatives (vega, vanna, volga, zomma, ultima) are reported per 1
  VOLATILITY POINT, so the raw derivative divided by 100 (or 100^2 for volga,
  100^3 for ultima).

Run:
    py -3 higher_order_greeks.py              # validate + real chain + positions
    py -3 higher_order_greeks.py --validate   # formula checks only, no network
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

SQ2PI = math.sqrt(2.0 * math.pi)
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton core
# --------------------------------------------------------------------------- #
def _phi(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQ2PI


def _N(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def d1d2(S, K, tau, r, q, sig):
    S, K, tau, sig = map(lambda z: np.asarray(z, dtype=float), (S, K, tau, sig))
    v = sig * np.sqrt(tau)
    a = (np.log(S / K) + (r - q + 0.5 * sig**2) * tau) / v
    return a, a - v


def bs_price(S, K, tau, r, q, sig, kind="call"):
    a, b = d1d2(S, K, tau, r, q, sig)
    df_q, df_r = np.exp(-q * tau), np.exp(-r * tau)
    if kind == "call":
        return S * df_q * _N(a) - K * df_r * _N(b)
    return K * df_r * _N(-b) - S * df_q * _N(-a)


# --------------------------------------------------------------------------- #
# Analytic greeks. Raw derivatives, no scaling. Scaling happens in report().
# --------------------------------------------------------------------------- #
def greeks(S, K, tau, r, q, sig, kind="call"):
    a, b = d1d2(S, K, tau, r, q, sig)
    rt = np.sqrt(tau)
    pa = _phi(a)
    eq, er = np.exp(-q * tau), np.exp(-r * tau)

    vega = S * eq * pa * rt
    gamma = eq * pa / (S * sig * rt)

    if kind == "call":
        delta = eq * _N(a)
        rho = K * tau * er * _N(b)
        theta = (-S * pa * sig * eq / (2 * rt)
                 + q * S * eq * _N(a) - r * K * er * _N(b))
        charm = (q * eq * _N(a)
                 - eq * pa * (2 * (r - q) * tau - b * sig * rt) / (2 * tau * sig * rt))
    else:
        delta = eq * (_N(a) - 1.0)
        rho = -K * tau * er * _N(-b)
        theta = (-S * pa * sig * eq / (2 * rt)
                 - q * S * eq * _N(-a) + r * K * er * _N(-b))
        charm = (-q * eq * _N(-a)
                 - eq * pa * (2 * (r - q) * tau - b * sig * rt) / (2 * tau * sig * rt))

    # second order
    vanna = -eq * pa * b / sig                       # dDelta/dsig = dVega/dS
    volga = vega * a * b / sig                       # dVega/dsig
    # published veta and color tables are written in tau, charm's is written in t.
    # negate both so every time derivative here is d/dt. Checked in validate().
    veta = S * eq * pa * rt * (q + (r - q) * a / (sig * rt) - (1 + a * b) / (2 * tau))

    # third order
    speed = -gamma / S * (a / (sig * rt) + 1.0)      # dGamma/dS
    zomma = gamma * (a * b - 1.0) / sig              # dGamma/dsig
    color = (eq * pa / (2 * S * tau * sig * rt)
             * (2 * q * tau + 1.0
                + (2 * (r - q) * tau - b * sig * rt) / (sig * rt) * a))
    ultima = -vega / sig**2 * (a * b * (1 - a * b) + a * a + b * b)

    return dict(d1=a, d2=b, price=bs_price(S, K, tau, r, q, sig, kind),
                delta=delta, vega=vega, theta=theta, rho=rho,
                gamma=gamma, vanna=vanna, volga=volga, charm=charm, veta=veta,
                speed=speed, zomma=zomma, color=color, ultima=ultima)


# --------------------------------------------------------------------------- #
# Finite-difference validation. Nothing goes on screen until this passes.
# --------------------------------------------------------------------------- #
def _fd(f, x, h, order=1):
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h**3)


def validate(verbose=True):
    """Differentiate the price surface numerically and compare to every formula."""
    cases = [
        dict(S=100.0, K=100.0, tau=0.25, r=0.042, q=0.005, sig=0.28, kind="call"),
        dict(S=100.0, K=110.0, tau=0.08, r=0.042, q=0.000, sig=0.45, kind="call"),
        dict(S=100.0, K=92.0, tau=1.00, r=0.042, q=0.020, sig=0.20, kind="put"),
        dict(S=180.0, K=175.0, tau=0.04, r=0.042, q=0.000, sig=0.55, kind="put"),
    ]
    rows, worst = [], 0.0
    for c in cases:
        S, K, tau, r, q, sig, kind = (c["S"], c["K"], c["tau"], c["r"],
                                      c["q"], c["sig"], c["kind"])
        g = greeks(**c)
        P = lambda s=S, k=K, t=tau, v=sig: float(bs_price(s, k, t, r, q, v, kind))
        hS, hV, hT = S * 2e-4, 1e-4, 1e-5

        num = {
            "delta": _fd(lambda s: P(s=s), S, hS),
            "vega": _fd(lambda v: P(v=v), sig, hV),
            "theta": -_fd(lambda t: P(t=t), tau, hT),
            "gamma": _fd(lambda s: P(s=s), S, hS, order=2),
            "vanna": (_fd(lambda v: _fd(lambda s: P(s=s, v=v), S, hS), sig, hV)),
            "volga": _fd(lambda v: P(v=v), sig, hV, order=2),
            "charm": -_fd(lambda t: float(greeks(S, K, t, r, q, sig, kind)["delta"]), tau, hT),
            "veta": -_fd(lambda t: float(greeks(S, K, t, r, q, sig, kind)["vega"]), tau, hT),
            # third derivative of price is roundoff-dominated, so also check the
            # cheaper chain: d/dS of an already-validated analytic gamma
            "speed": _fd(lambda s: float(greeks(s, K, tau, r, q, sig, kind)["gamma"]), S, hS),
            "zomma": _fd(lambda v: float(greeks(S, K, tau, r, q, v, kind)["gamma"]), sig, hV),
            "color": -_fd(lambda t: float(greeks(S, K, t, r, q, sig, kind)["gamma"]), tau, hT),
            "ultima": _fd(lambda v: float(greeks(S, K, tau, r, q, v, kind)["volga"]), sig, hV),
        }
        for name, nv in num.items():
            av = float(g[name])
            scale = max(abs(av), abs(nv), 1e-12)
            err = abs(av - nv) / scale
            worst = max(worst, err)
            rows.append((f"{kind[0].upper()} K{K:g} T{tau:g}", name, av, nv, err))

    if verbose:
        print("\nFORMULA VALIDATION  (analytic vs finite difference of the price surface)")
        print(f"{'case':<16}{'greek':<9}{'analytic':>15}{'numeric':>15}{'rel err':>11}")
        for case, name, av, nv, err in rows:
            flag = "" if err < 2e-4 else "   <-- CHECK"
            print(f"{case:<16}{name:<9}{av:>15.6g}{nv:>15.6g}{err:>11.2e}{flag}")
        print(f"\nworst relative error across {len(rows)} checks: {worst:.2e}")
    return worst


# --------------------------------------------------------------------------- #
# Real market data
# --------------------------------------------------------------------------- #
def _safe_int(v):
    try:
        f = float(v)
        return 0 if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def fetch_chain(ticker="NVDA", max_expiries=6):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["last_price"])
    # yfinance's info["dividendYield"] flips between fraction and percent and has
    # returned 50.0 for NVDA, so build q from the actual trailing cash dividends.
    try:
        div = tk.dividends
        cutoff = div.index.max() - dt.timedelta(days=365)
        q = float(div[div.index > cutoff].sum()) / spot
        if not (0.0 <= q < 0.25):
            q = 0.0
    except Exception:
        q = 0.0

    try:
        r = float(yf.Ticker("^IRX").fast_info["last_price"]) / 100.0
    except Exception:
        r = 0.042

    today = dt.date.today()
    out = []
    for exp in tk.options[:max_expiries]:
        d = dt.date.fromisoformat(exp)
        tau = (d - today).days / 365.0
        if tau <= 0.003:
            continue
        ch = tk.option_chain(exp)
        for kind, df in (("call", ch.calls), ("put", ch.puts)):
            for _, row in df.iterrows():
                iv = float(row.get("impliedVolatility") or 0.0)
                bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
                if not (0.01 < iv < 4.0) or bid <= 0 or ask <= 0:
                    continue
                out.append(dict(expiry=exp, tau=tau, kind=kind,
                                K=float(row["strike"]), iv=iv,
                                mid=0.5 * (bid + ask), bid=bid, ask=ask,
                                oi=_safe_int(row.get("openInterest")),
                                vol=_safe_int(row.get("volume"))))
    return dict(ticker=ticker, spot=spot, r=r, q=q, asof=str(today), rows=out)


def nearest(rows, tau_target, kind, delta_target, spot, r, q):
    """Pick the listed contract whose BS delta is closest to a target."""
    taus = sorted({x["tau"] for x in rows}, key=lambda t: abs(t - tau_target))
    tau = taus[0]
    pool = [x for x in rows if x["tau"] == tau and x["kind"] == kind]
    best, bd = None, 9e9
    for x in pool:
        d = float(greeks(spot, x["K"], x["tau"], r, q, x["iv"], kind)["delta"])
        if abs(d - delta_target) < bd:
            best, bd = dict(x, delta=d), abs(d - delta_target)
    return best


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
SCALE = {                      # raw derivative -> the units a desk quotes
    "delta": (1.0, "per $1"),
    "gamma": (1.0, "delta per $1"),
    "vega": (0.01, "per 1 vol pt"),
    "theta": (1 / 365.0, "per day"),
    "vanna": (0.01, "delta per 1 vol pt"),
    "volga": (0.0001, "vega per 1 vol pt"),
    "charm": (1 / 365.0, "delta per day"),
    "veta": (0.01 / 365.0, "vega per day"),
    "speed": (1.0, "gamma per $1"),
    "zomma": (0.01, "gamma per 1 vol pt"),
    "color": (1 / 365.0, "gamma per day"),
    "ultima": (1e-6, "volga per 1 vol pt"),
}
ORDER = {"delta": 1, "vega": 1, "theta": 1,
         "gamma": 2, "vanna": 2, "volga": 2, "charm": 2, "veta": 2,
         "speed": 3, "zomma": 3, "color": 3, "ultima": 3}


def report(tag, S, K, tau, r, q, sig, kind, mult=1.0):
    g = greeks(S, K, tau, r, q, sig, kind)
    print(f"\n{tag}")
    print(f"  {kind} K={K:g}  S={S:.2f}  {tau*365:.0f} days  IV={sig*100:.1f}%  "
          f"price={float(g['price']):.2f}  d1={float(g['d1']):+.3f}")
    for name, (sc, unit) in SCALE.items():
        print(f"    {ORDER[name]}  {name:<7}{float(g[name])*sc*mult:>13.5f}   {unit}")
    return g


def taylor_attribution(S, K, tau, r, q, sig, kind, dS, dsig, days):
    """How much of a real repricing does each order of the expansion explain."""
    g = greeks(S, K, tau, r, q, sig, kind)
    dt_yr = days / 365.0
    exact = float(bs_price(S + dS, K, tau - dt_yr, r, q, sig + dsig, kind)) - float(g["price"])

    t1 = float(g["delta"]) * dS + float(g["vega"]) * dsig + float(g["theta"]) * dt_yr
    t2 = (0.5 * float(g["gamma"]) * dS**2
          + 0.5 * float(g["volga"]) * dsig**2
          + float(g["vanna"]) * dS * dsig
          + float(g["charm"]) * dS * dt_yr
          + float(g["veta"]) * dsig * dt_yr)
    t3 = ((1 / 6) * float(g["speed"]) * dS**3
          + 0.5 * float(g["zomma"]) * dS**2 * dsig
          + 0.5 * float(g["color"]) * dS**2 * dt_yr
          + (1 / 6) * float(g["ultima"]) * dsig**3)
    return dict(exact=exact, o1=t1, o2=t1 + t2, o3=t1 + t2 + t3,
                d_o1=t1, d_o2=t2, d_o3=t3,
                err1=exact - t1, err2=exact - (t1 + t2), err3=exact - (t1 + t2 + t3))


def skew_slope(rows, tau, S, r, q):
    """dIV/dK around the money from the real chain, then converted to dIV/dS.

    This is what lets the P&L scenario use a vol shock the market itself quotes
    rather than one invented for the video.
    """
    pool = sorted([x for x in rows if abs(x["tau"] - tau) < 1e-9],
                  key=lambda x: x["K"])
    band = [x for x in pool if 0.90 * S <= x["K"] <= 1.10 * S]
    if len(band) < 6:
        return None
    ks = np.array([x["K"] for x in band], float)
    ivs = np.array([x["iv"] for x in band], float)
    slope = float(np.polyfit(ks, ivs, 1)[0])          # dIV per $1 of strike
    return slope


def chain_profile(rows, tau, S, r, q, kind="call"):
    """Every greek across the real strikes of one real expiry."""
    pool = sorted([x for x in rows if abs(x["tau"] - tau) < 1e-9 and x["kind"] == kind],
                  key=lambda x: x["K"])
    out = []
    for x in pool:
        g = greeks(S, x["K"], tau, r, q, x["iv"], kind)
        # desk units, same scaling as the position table, so nothing quoted in
        # the video mixes per-year with per-day or per-unit-vol with per-vol-point
        out.append(dict(K=x["K"], iv=x["iv"], oi=x["oi"], mid=x["mid"],
                        **{k: float(g[k]) * SCALE[k][0] for k in SCALE}))
    return out


def peak_of(profile, name, absolute=True):
    if not profile:
        return None
    key = (lambda p: abs(p[name])) if absolute else (lambda p: p[name])
    return max(profile, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    worst = validate()
    if worst > 2e-4:
        print("\nVALIDATION FAILED, refusing to produce numbers", file=sys.stderr)
        return 1
    if a.validate:
        return 0

    ch = fetch_chain(a.ticker, max_expiries=14)
    S, r, q, rows = ch["spot"], ch["r"], ch["q"], ch["rows"]
    print(f"\n\n{'='*78}\nREAL CHAIN  {ch['ticker']}  spot {S:.2f}  r {r*100:.2f}%  "
          f"q {q*100:.2f}%  as of {ch['asof']}  ({len(rows)} live contracts)\n{'='*78}")

    taus = sorted({x["tau"] for x in rows})
    tau30 = min(taus, key=lambda t: abs(t - 30 / 365))
    tau7 = min(taus, key=lambda t: abs(t - 7 / 365))
    print(f"expiries in use: {tau30*365:.0f} days and {tau7*365:.0f} days")

    # ---- 1. where each greek actually peaks on a real chain ----------------
    prof = chain_profile(rows, tau30, S, r, q, "call")
    atm = min(prof, key=lambda p: abs(p["K"] - S))
    print(f"\n--- {tau30*365:.0f}-day calls, {len(prof)} live strikes, "
          f"ATM strike {atm['K']:g} at IV {atm['iv']*100:.1f}% ---")
    print(f"{'greek':<8}{'peaks at K':>12}{'moneyness':>11}{'value there':>14}"
          f"{'ATM value':>13}")
    sd = S * atm["iv"] * math.sqrt(tau30)
    print(f"(one 30-day standard deviation is ${sd:.2f}, so 1sd out is "
          f"K={S+sd:.0f})")
    for name in ("gamma", "vanna", "volga", "charm", "speed", "zomma", "color"):
        p = peak_of(prof, name)
        print(f"{name:<8}{p['K']:>12g}{p['K']/S:>11.3f}{p[name]:>14.5f}"
              f"{atm[name]:>13.5f}   ({(p['K']-S)/sd:+.2f} sd)")

    # how the same greeks scale into expiry: 30 days versus 7 days, ATM
    prof7 = chain_profile(rows, tau7, S, r, q, "call")
    if prof7:
        atm7 = min(prof7, key=lambda p: abs(p["K"] - S))
        print(f"\nATM, {tau30*365:.0f} days versus {tau7*365:.0f} days "
              f"(K {atm['K']:g} and {atm7['K']:g}, IV {atm['iv']*100:.1f}% "
              f"and {atm7['iv']*100:.1f}%):")
        print(f"{'greek':<8}{'30d':>13}{'7d':>13}{'ratio':>9}")
        for name in ("gamma", "vega", "theta", "vanna", "volga", "charm",
                     "speed", "zomma", "color"):
            a30, a7 = atm[name], atm7[name]
            rr = (a7 / a30) if abs(a30) > 1e-12 else float("nan")
            print(f"{name:<8}{a30:>13.5f}{a7:>13.5f}{rr:>9.2f}x")

    # ---- 2. three positions, each one isolating a greek --------------------
    print("\n\n--- POSITIONS BUILT FROM LIVE CONTRACTS (1 lot = 100 shares) ---")
    c25 = nearest(rows, tau30, "call", 0.25, S, r, q)
    p25 = nearest(rows, tau30, "put", -0.25, S, r, q)
    catm = nearest(rows, tau30, "call", 0.50, S, r, q)
    patm = nearest(rows, tau30, "put", -0.50, S, r, q)

    def leg(x, kind, n):
        g = greeks(S, x["K"], x["tau"], r, q, x["iv"], kind)
        return {k: float(g[k]) * SCALE[k][0] * n * 100 for k in SCALE}

    books = {
        "short ATM straddle": [(catm, "call", -1), (patm, "put", -1)],
        "25d risk reversal (long call, short put)": [(c25, "call", 1), (p25, "put", -1)],
        "25d strangle (long both wings)": [(c25, "call", 1), (p25, "put", 1)],
    }
    print(f"{'book':<42}" + "".join(f"{k:>10}" for k in
          ("delta", "gamma", "vega", "theta", "vanna", "volga", "charm", "speed", "zomma")))
    book_greeks = {}
    for label, legs in books.items():
        tot = {k: 0.0 for k in SCALE}
        for x, kind, n in legs:
            for k, v in leg(x, kind, n).items():
                tot[k] += v
        book_greeks[label] = tot
        print(f"{label:<42}" + "".join(f"{tot[k]:>10.2f}" for k in
              ("delta", "gamma", "vega", "theta", "vanna", "volga", "charm", "speed", "zomma")))
    print(f"\nstrikes used: ATM call {catm['K']:g} / put {patm['K']:g}, "
          f"25d call {c25['K']:g} (IV {c25['iv']*100:.1f}%) / "
          f"25d put {p25['K']:g} (IV {p25['iv']*100:.1f}%)")
    print(f"25-delta risk reversal skew: put IV minus call IV = "
          f"{(p25['iv']-c25['iv'])*100:+.2f} vol points")

    # ---- 3. P&L attribution on a real move, real skew-implied vol shock ----
    import yfinance as yf
    hist = yf.Ticker(a.ticker).history(period="1y")["Close"]
    rets = np.log(hist / hist.shift(1)).dropna()
    worst_day = float(rets.min())
    realized = float(rets.std() * math.sqrt(252))
    slope = skew_slope(rows, tau30, S, r, q)
    print(f"\n\n--- P&L ATTRIBUTION ON A REAL MOVE ---")
    print(f"{a.ticker} worst 1-day log return in the last year: {worst_day*100:.2f}%")
    print(f"1y realized vol {realized*100:.1f}%, 30d implied {atm['iv']*100:.1f}%, "
          f"premium {(atm['iv']-realized)*100:+.1f} vol points")
    print(f"chain skew slope near the money: {slope*100:+.4f} vol pts per $1 of strike")

    dS = S * (math.exp(worst_day) - 1.0)
    dsig = -slope * dS                       # slide down the quoted smile
    print(f"scenario: dS = {dS:+.2f} ({worst_day*100:.2f}%), "
          f"dIV = {dsig*100:+.2f} vol pts read off the live skew, 1 day of decay\n")

    header = (f"{'contract':<24}{'exact':>10}{'1st ord':>10}{'+2nd':>10}"
              f"{'+3rd':>10}{'err 1':>9}{'err 2':>9}{'err 3':>9}")
    print(header)
    attrib = {}
    for tag, x, kind in (("ATM call", catm, "call"),
                         ("25d call", c25, "call"),
                         ("25d put", p25, "put")):
        t = taylor_attribution(S, x["K"], tau30, r, q, x["iv"], kind, dS, dsig, 1.0)
        attrib[tag] = t
        print(f"{tag+' K'+format(x['K'],'g'):<24}"
              f"{t['exact']:>10.3f}{t['o1']:>10.3f}{t['o2']:>10.3f}{t['o3']:>10.3f}"
              f"{t['err1']:>9.3f}{t['err2']:>9.3f}{t['err3']:>9.3f}")

    print("\nsame thing as a share of the true price change:")
    print(f"{'contract':<24}{'1st ord':>10}{'+2nd':>10}{'+3rd':>10}")
    for tag, t in attrib.items():
        e = t["exact"]
        print(f"{tag:<24}{t['o1']/e*100:>9.1f}%{t['o2']/e*100:>9.1f}%{t['o3']/e*100:>9.1f}%")

    # ---- 4. market wide vanna and charm from real open interest -----------
    print("\n\n--- OPEN INTEREST WEIGHTED EXPOSURE, WHOLE LISTED CHAIN ---")
    print("assumes dealers are short every open contract, the standard sign "
          "assumption behind GEX style numbers and the weakest link in it")
    agg = {k: 0.0 for k in ("gamma", "vanna", "charm", "volga", "zomma", "speed")}
    n_used = 0
    for x in rows:
        if x["oi"] <= 0:
            continue
        g = greeks(S, x["K"], x["tau"], r, q, x["iv"], x["kind"])
        for k in agg:
            agg[k] += -float(g[k]) * SCALE[k][0] * x["oi"] * 100
        n_used += 1
    print(f"contracts with open interest: {n_used}")
    print(f"  dealer gamma  {agg['gamma']*S/100:>16,.0f} shares per 1% move")
    print(f"  dealer vanna  {agg['vanna']:>16,.0f} shares of delta per 1 vol pt")
    print(f"  dealer charm  {agg['charm']:>16,.0f} shares of delta per day")
    print(f"  dealer volga  {agg['volga']:>16,.0f} vega per 1 vol pt")
    print(f"  dealer zomma  {agg['zomma']*S/100:>16,.0f} gamma-shares per 1 vol pt")

    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(meta=dict(ticker=ch["ticker"], spot=S, r=r, q=q, asof=ch["asof"],
                           tau30=tau30, tau7=tau7, realized=realized,
                           worst_day=worst_day, skew_slope=slope, dS=dS, dsig=dsig),
                 profile=prof, books=book_greeks, attribution=attrib, aggregate=agg),
            indent=1, default=float), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
