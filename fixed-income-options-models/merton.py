"""4. Merton (1974): a company's shares are a call on its assets, struck at its debt.

Market cap and equity vol come from Yahoo, debt and shares from SEC filings. The two
equations

    E = V N(d1) - D exp(-rT) N(d2)        sigma_E E = N(d1) sigma_V V

are solved for the asset value V and its vol sigma_V, with the KMV default point
D = debt due within a year + half the long term debt, and T = 1 year. Distance to default
uses the risk free drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

import style as s
from curve import TreasuryCurve
from data import debt_and_shares, fred, prices

COMPANIES = {"BA": (12927, "Boeing"), "CCL": (815097, "Carnival"), "INTC": (50863, "Intel"),
             "T": (732717, "AT&T"), "AAPL": (320193, "Apple")}


def solve(E, sE, D, r, T=1.0):
    """Asset value, asset vol and distance to default from equity value and equity vol."""
    def gap(z):
        V, sV = z
        sV = abs(sV)
        d1 = (np.log(V / D) + (r + sV ** 2 / 2) * T) / (sV * np.sqrt(T))
        d2 = d1 - sV * np.sqrt(T)
        return [(V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E) / E,
                (norm.cdf(d1) * sV * V - sE * E) / E]
    V, sV = fsolve(gap, [E + D, sE * E / (E + D)], xtol=1e-10)
    sV = abs(sV)
    dd = (np.log(V / D) + (r - sV ** 2 / 2) * T) / (sV * np.sqrt(T))
    return float(V), float(sV), float(dd)


def today(tk, curve: TreasuryCurve, r):
    debt, shares = debt_and_shares(tk, COMPANIES[tk][0])
    px = prices(tk, "2017-01-01", adjusted=False).loc[:curve.date]
    D = float(debt["short"].iloc[-1] + 0.5 * debt["long"].iloc[-1])
    E = float(px.iloc[-1] * shares.iloc[-1])
    sE = float(np.log(px).diff().dropna().iloc[-252:].std() * np.sqrt(252))
    V, sV, dd = solve(E, sE, D, r)
    return dict(equity=E, equity_vol=sE * 100, default_point=D, total_debt=float(debt.iloc[-1].sum()),
                assets=V, asset_vol=sV * 100, distance_to_default=dd, price=float(px.iloc[-1]),
                balance_sheet=str(debt.index[-1].date()))


def run(curve: TreasuryCurve):
    r1 = fred("DGS1", "2017-01-01").loc[:curve.date] / 100
    r = float(r1.iloc[-1])
    now = {tk: today(tk, curve, r) for tk in COMPANIES}

    # Boeing through time, weekly: shares and debt carried forward from each quarter end
    debt, shares = debt_and_shares("BA", COMPANIES["BA"][0])
    px = prices("BA", "2017-01-01", adjusted=False).loc[:curve.date]
    sE = np.log(px).diff().rolling(252).std() * np.sqrt(252)
    rows = []
    for d in px.loc["2018-01-01":].index[::5]:
        dv, sh = debt.loc[:d], shares.loc[:d]
        if not len(dv) or not len(sh) or np.isnan(sE.loc[d]):
            continue
        D = float(dv["short"].iloc[-1] + 0.5 * dv["long"].iloc[-1])
        V, sV, dd = solve(float(px.loc[d] * sh.iloc[-1]), float(sE.loc[d]), D, float(r1.loc[:d].iloc[-1]))
        rows.append((d, dd))
    hist = pd.DataFrame(rows, columns=["date", "dd"]).set_index("date")

    b = now["BA"]
    fig, ax = s.figure(6.2, 4.4)
    D, sV = b["default_point"], b["asset_vol"] / 100
    Vs = np.linspace(0.02, 4.0 * D, 300)                  # zoomed to where the option bends
    d1 = (np.log(Vs / D) + (r + sV ** 2 / 2)) / sV
    E_ = Vs * norm.cdf(d1) - D * np.exp(-r) * norm.cdf(d1 - sV)
    ax.plot(Vs, np.maximum(Vs - D, 0), color=s.INK, lw=1.2, ls=(0, (4, 3)), alpha=0.75, label="payoff at the horizon")
    ax.plot(Vs, E_, color=s.GREEN, lw=2.6, label="shares = call on the assets")
    ax.plot(Vs, Vs - E_, color=s.PINK, lw=2.2, label="debt = safe bond minus a put")
    ax.axvline(D, color=s.GRID, lw=1.0)
    ax.text(D * 1.05, max(E_) * 0.55, f"default point\n${D:.0f}bn of debt", color=s.INK, fontsize=8.3, va="top")
    ax.text(0.97, 0.05, f"today, off the chart to the right:\nassets \\${b['assets']:.0f}bn, shares "
            f"\\${b['equity']:.0f}bn,\n{b['distance_to_default']:.1f} std devs from default", transform=ax.transAxes,
            ha="right", va="bottom", color=s.GREEN, fontsize=8.3, fontweight="bold")
    s.labels(ax, "Boeing asset value ($bn)", "value ($bn)", "Merton: Boeing's shares are a call option")
    s.legend(ax, loc="upper left")
    ax.set_xlim(0, Vs[-1])
    s.save(fig, "merton_call.png")

    fig, ax = s.figure(6.2, 4.0)
    ax.plot(hist.index, hist["dd"], color=s.GREEN, lw=2.0)
    for date, txt, (dx, dy, ha) in (("2019-03-13", "737 MAX grounded", (34, 30, "left")),
                                    ("2020-03-20", "Covid", (-36, -6, "right")),
                                    ("2024-01-05", "door plug", (0, 30, "center"))):
        y_ = float(hist["dd"].loc[:pd.Timestamp(date)].iloc[-1])
        ax.annotate(txt, (pd.Timestamp(date), y_), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    color=s.INK, fontsize=8.2, arrowprops=dict(arrowstyle="-", color=s.INK, lw=0.6))
    ax.set_ylim(0, hist["dd"].max() * 1.12)
    s.labels(ax, None, "distance to default (std devs)", "Boeing, distance to default, weekly")
    s.save(fig, "merton_boeing.png")

    fig, ax = s.figure(6.2, 4.0, light=True)
    order = sorted(now, key=lambda k: now[k]["distance_to_default"])
    vals = [now[k]["distance_to_default"] for k in order]
    ax.barh([COMPANIES[k][1] for k in order], vals, color=[s.ORANGE if k == "BA" else s.INDIGO for k in order], alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.1f}", va="center", color=s.LIGHT_INK, fontsize=8.6, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="y", visible=False)
    s.labels(ax, "distance to default (std devs)", None, f"Same model, five companies, {curve.date:%d %b %Y}", light=True)
    s.save(fig, "merton_five.png", light=True)

    low = hist["dd"].idxmin()
    facts = dict(r=r * 100, today=now, boeing_low_dd=float(hist["dd"].min()), boeing_low_date=str(low.date()),
                 since=str(hist.index[0].date()))
    s.record("merton", facts)
    return facts
