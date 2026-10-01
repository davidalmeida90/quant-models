"""Same day SPX gamma closing in on expiration, step by step.

Pulls the full SPX chain from Cboe's free delayed quotes and the SqueezeMetrics daily GEX history, then:
  1. ages the same day expiry hour by hour (spot, vols and open interest held fixed) and re-prices dealer
     gamma across SPX levels, so the gamma flip can be followed into the close
  2. splits the same day book by strike
  3. sizes the hedge a 0.25% move forces at the biggest same day strike, next to the usual 1% figure
  4. lines up every SqueezeMetrics day against the next day's SPX move
  5. measures how much the dealer sign convention moves the answer

Writes four charts to charts/ and the numbers to results.json. Both sources are read at run time, nothing
is cached or redistributed. Run it during US market hours for a live same day expiry.

    python gex_0dte.py
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

# ---- step 1: the chain, with the same day expiry marked ---------------------------------------------
CHAIN = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<ymd>\d{6})(?P<side>[CP])(?P<strike>\d{8})$")
MULTIPLIER = 100          # shares per SPX contract
YEAR_H = 365 * 24         # vols are annualised on calendar time, so hours go over 8,760


def fetch_chain():
    """Every SPX and SPXW contract with open interest. same_day marks the expiry that settles today."""
    raw = json.loads(urllib.request.urlopen(
        urllib.request.Request(CHAIN, headers={"User-Agent": "gex-research"}), timeout=90).read())
    stamp = raw.get("timestamp") or datetime.now().isoformat(timespec="minutes")
    today = date.fromisoformat(stamp[:10])
    rows = []
    for o in raw["data"]["options"]:
        m = OCC.match(o["option"])
        if m:
            rows.append(dict(expiry=date(2000 + int(m["ymd"][:2]), int(m["ymd"][2:4]), int(m["ymd"][4:])),
                             is_call=m["side"] == "C", strike=int(m["strike"]) / 1000,
                             oi=float(o["open_interest"]), iv=float(o["iv"])))
    df = pd.DataFrame(rows)
    spot = float(raw["data"]["current_price"])
    df["days"] = [(e - today).days for e in df.expiry]
    df = df[(df.days >= 0) & (df.oi > 0) & (np.abs(np.log(df.strike / spot)) < 0.5)].reset_index(drop=True)
    df["same_day"] = df.days == 0
    return df, spot, stamp


def same_day_vols(df, spot):
    """Same day quotes with no usable vol borrow the next expiry's vol at the same strike and side, else ATM."""
    nxt = df[df.days == df.loc[~df.same_day, "days"].min()]
    lookup = {(k, c): v for k, c, v in zip(nxt.strike, nxt.is_call, nxt.iv)}
    atm = float(nxt.loc[(nxt.strike - spot).abs().idxmin(), "iv"])
    iv = df.iv.to_numpy().copy()
    bad = df.same_day.to_numpy() & (iv < 0.02)
    iv[bad] = [lookup.get((k, c), atm) for k, c in zip(df.strike[bad], df.is_call[bad])]
    return iv, atm


# ---- step 2: dealer gamma, with the same day expiry aged ---------------------------------------------
def gamma(S, K, T, sigma):
    """Black-Scholes gamma (rate set to zero, it barely moves gamma over hours)."""
    srt = np.maximum(sigma, 1e-4) * np.sqrt(np.maximum(T, 0.05 / YEAR_H))
    d1 = np.log(S / K) / srt + 0.5 * srt
    return np.exp(-0.5 * d1 ** 2) / (np.sqrt(2 * np.pi) * S * srt)


def dealer_gex(df, iv, S, hours, sign=None):
    """$ dealers trade per 1% SPX move, per contract. Default sign: dealers long calls, short puts.
    Same day contracts get `hours` to expiration, every other expiry keeps its days."""
    T = np.where(df.same_day, hours / YEAR_H, df.days / 365)
    sign = np.where(df.is_call, 1.0, -1.0) if sign is None else sign
    return sign * gamma(S, df.strike.to_numpy(), T, iv) * df.oi.to_numpy() * MULTIPLIER * S * S * 0.01


def profile(df, iv, grid, hours, sign=None):
    """Net dealer gamma at every hypothetical SPX level: re-price the whole book, add it up."""
    return np.array([dealer_gex(df, iv, S, hours, sign).sum() for S in grid])


def flip(grid, total, spot):
    """Zero crossing of the profile closest to spot."""
    best = None
    for i in range(len(grid) - 1):
        a, b = total[i], total[i + 1]
        if (a < 0 <= b) or (a > 0 >= b):
            x = grid[i] + (grid[i + 1] - grid[i]) * (-a) / (b - a)
            best = x if best is None or abs(x - spot) < abs(best - spot) else best
    return best


# ---- step 3: the same day book, strike by strike ----------------------------------------------------
def same_day_by_strike(df, iv, spot, hours):
    sd = df.same_day.to_numpy()
    x = pd.Series(dealer_gex(df, iv, spot, hours)[sd]).groupby(df.strike[sd].to_numpy()).sum()
    return x / 1e9


# ---- step 4: the hedge a small move forces, next to the 1% figure -----------------------------------
MOVE = 0.0025             # a 0.25% move, about what SPX does in a last hour


def hedge_for_move(df, iv, K, hours, move=MOVE):
    """$ of SPX dealers must trade at strike K if SPX moves `move` off it: the exact change in delta.
    It can never pass the strike's whole delta, contracts x 100 x K."""
    m = (df.same_day & (df.strike == K)).to_numpy()
    srt = iv[m] * np.sqrt(hours / YEAR_H)
    d_on, d_off = 0.5 * srt, np.log(1 + move) / srt + 0.5 * srt
    return float(((norm.cdf(d_off) - norm.cdf(d_on)) * df.oi.to_numpy()[m] * MULTIPLIER * K).sum())


def linear_one_pct(df, iv, K, hours):
    """The usual GEX figure at the strike: gamma x S squared x 1%. Fine far from expiry, it explodes close to it."""
    m = (df.same_day & (df.strike == K)).to_numpy()
    return float((gamma(K, K, hours / YEAR_H, iv[m]) * df.oi.to_numpy()[m] * MULTIPLIER * K * K * 0.01).sum())


# ---- step 5: fifteen years of dealer gamma against the next day -------------------------------------
SQUEEZE = "https://squeezemetrics.com/monitor/static/DIX.csv"


def squeeze_history():
    """SqueezeMetrics GEX (their own positioning estimate, their own scale) and the next close to close SPX move."""
    raw = urllib.request.urlopen(urllib.request.Request(SQUEEZE, headers={"User-Agent": "gex-research"}),
                                 timeout=60).read().decode()
    sq = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
    sq["next"] = sq.price.pct_change().shift(-1) * 100
    sq["gex_bn"] = sq.gex / 1e9
    return sq.dropna(subset=["next"]).reset_index(drop=True)


def spreads(sq):
    short, long_ = sq[sq.gex < 0], sq[sq.gex > 0]
    calm = short[~short.date.dt.year.isin([2020, 2022])]
    return dict(days=len(sq), short_days=len(short), short=float(short.next.std()), long=float(long_.next.std()),
                short_without_2020_2022=float(calm.next.std()))


# ---- step 6: how much the sign convention moves the answer ------------------------------------------
def sign_scenarios(df, iv, spot, hours):
    """Dealer gamma at spot under four views of who holds the same day book."""
    std = np.where(df.is_call, 1.0, -1.0)
    sd = df.same_day.to_numpy()
    netted = np.where(sd, 0.0, std)                           # same day flow two way, nets to zero
    puts_sold = np.where(sd & ~df.is_call.to_numpy(), 1.0, std)  # customers sold the same day puts
    return {name: float(dealer_gex(df, iv, spot, hours, s).sum() / 1e9) for name, s in
            (("convention", None), ("same_day_nets_to_zero", netted), ("same_day_puts_sold", puts_sold))}


# ---- charts ----------------------------------------------------------------------------------------
NAVY, BLUE, RED, GOLD, GREY = "#191936", "#3C6E9E", "#C4473F", "#D4A017", "#8a93a3"
HOURS = (6.5, 3.0, 1.0, 0.25, 0.1)
SHADES = ("#b9c6d8", "#8aa1bf", "#5a7ca6", "#2f4f80", NAVY)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9cfdb"); ax.spines["bottom"].set_color("#c9cfdb")
    ax.tick_params(colors="#4a5361", labelsize=10)
    ax.grid(axis="y", color="#eef0f3", lw=1)
    ax.set_axisbelow(True)


def chart_aging(grid, curves, flips, spot, path):
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=110)
    for h, c, col in zip(HOURS, curves, SHADES):
        ax.plot(grid, c / 1e9, color=col, lw=2.2 if h == HOURS[-1] else 1.6, label=f"{h:g}h to expiration")
    ax.axhline(0, color=GREY, lw=1)
    ax.axvline(spot, color=GOLD, lw=1.6)
    ax.text(spot, ax.get_ylim()[1] * 0.92, f" spot {spot:,.0f}", color="#9A7412", fontsize=10, fontweight="bold")
    for h, f, col in zip(HOURS, flips, SHADES):
        if f:
            ax.plot([f], [0], "o", color=col, ms=6)
    ok = [f for f in flips if f]
    if ok:
        ax.annotate(f"gamma flip {ok[0]:,.0f} to {ok[-1]:,.0f}", (ok[-1], 0), xytext=(12, -24), textcoords="offset points",
                    color=NAVY, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("dealer gamma, $ bn per 1% move", color="#4a5361")
    ax.set_xlabel("hypothetical SPX level", color="#4a5361")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_strikes(first, last, spot, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110, sharey=True)
    for ax, s, h in zip(axes, (first, last), (HOURS[0], HOURS[-1])):
        s = s[(s.index > spot * 0.985) & (s.index < spot * 1.015)]
        ax.bar(s.index, s.values, width=3.6, color=np.where(s.values >= 0, BLUE, RED))
        ax.axhline(0, color=GREY, lw=1)
        ax.axvline(spot, color=GOLD, lw=1.4)
        ax.set_title(f"{h:g}h to expiration", color=NAVY, fontsize=11, loc="left")
        ax.set_xlabel("strike", color="#4a5361")
        style(ax)
    axes[0].set_ylabel("same day dealer gamma, $ bn per 1%", color="#4a5361")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_hedge(hh, linear, exact, cap, K, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110)
    axes[0].plot(hh, linear / 1e9, color=RED, lw=2.2)
    axes[0].set_title(f"usual figure at {K:,.0f}: gamma x 1% move", color=NAVY, fontsize=11, loc="left")
    axes[1].plot(hh, exact / 1e9, color=BLUE, lw=2.2)
    axes[1].axhline(cap / 1e9, color=GREY, lw=1, ls="--")
    axes[1].text(hh[0], cap / 1e9 * 0.95, f" whole strike: ${cap / 1e9:.1f}bn", color="#4a5361", fontsize=9.5, va="top")
    axes[1].set_ylim(0, cap / 1e9 * 1.08)
    axes[1].set_title(f"exact hedge for a {MOVE:.2%} move off {K:,.0f}", color=NAVY, fontsize=11, loc="left")
    for ax in axes:
        ax.set_xlim(hh[0], 0)
        ax.set_xlabel("hours to expiration", color="#4a5361")
        ax.set_ylabel("$ bn of SPX", color="#4a5361")
        style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_squeeze(sq, st, path):
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=110)
    neg = sq.gex < 0
    ax.scatter(sq.gex_bn[~neg], sq.next[~neg], s=7, color=BLUE, alpha=0.45, lw=0)
    ax.scatter(sq.gex_bn[neg], sq.next[neg], s=9, color=RED, alpha=0.7, lw=0)
    ax.axvline(0, color=GOLD, lw=1.4)
    for v, col, lo, hi in ((st["short"], RED, sq.gex_bn.min(), 0), (st["long"], BLUE, 0, sq.gex_bn.max())):
        ax.plot([lo, hi], [v, v], color=col, lw=1.6, ls="--"); ax.plot([lo, hi], [-v, -v], color=col, lw=1.6, ls="--")
    ax.text(0.2, ax.get_ylim()[0] * 0.92, "gamma flip", color="#9A7412", fontsize=10, fontweight="bold")
    ax.set_xlabel("SqueezeMetrics GEX that day, $ bn", color="#4a5361")
    ax.set_ylabel("next day SPX move, %", color="#4a5361")
    style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    df, spot, stamp = fetch_chain()
    iv, atm = same_day_vols(df, spot)
    if not df.same_day.any():
        raise SystemExit("no same day expiry in this chain: run it on a US trading day")
    grid = np.linspace(spot * 0.97, spot * 1.03, 121)
    curves = [profile(df, iv, grid, h) for h in HOURS]
    flips = [flip(grid, c, spot) for c in curves]
    at_spot = [float(np.interp(spot, grid, c) / 1e9) for c in curves]
    first, last = same_day_by_strike(df, iv, spot, HOURS[0]), same_day_by_strike(df, iv, spot, HOURS[-1])
    K = float(first.abs().idxmax())                      # the biggest same day strike at the open
    hh = np.linspace(HOURS[0], HOURS[-1], 120)
    linear = np.array([linear_one_pct(df, iv, K, h) for h in hh])
    exact = np.array([hedge_for_move(df, iv, K, h) for h in hh])
    m = (df.same_day & (df.strike == K)).to_numpy()
    cap = float(df.oi.to_numpy()[m].sum() * MULTIPLIER * K)
    sq = squeeze_history()
    st = spreads(sq)
    scen = sign_scenarios(df, iv, spot, HOURS[0])
    chart_aging(grid, curves, flips, spot, CHARTS / "gamma_flip_aging.png")
    chart_strikes(first, last, spot, CHARTS / "same_day_by_strike.png")
    chart_hedge(hh, linear, exact, cap, K, CHARTS / "hedge_vs_linear.png")
    chart_squeeze(sq, st, CHARTS / "gex_next_day.png")
    res = dict(stamp=stamp, spot=spot, contracts=len(df), same_day_contracts=int(df.same_day.sum()),
               same_day_oi=float(df.oi[df.same_day].sum()), atm_iv=atm,
               hours=list(HOURS), flips=flips, gex_at_spot_bn=at_spot,
               strike=K, strike_contracts=float(df.oi.to_numpy()[m].sum()), strike_cap_bn=cap / 1e9,
               linear_bn=[float(linear[0] / 1e9), float(linear[-1] / 1e9)],
               hedge_bn=[float(exact[0] / 1e9), float(exact[-1] / 1e9)],
               squeeze=st, squeeze_last_date=str(sq.date.iloc[-1].date()), sign_scenarios_bn=scen)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
