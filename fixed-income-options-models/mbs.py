"""3. Mortgage-backed security prepayment option.

Every US homeowner can repay at par any day, so an MBS holder is short thousands of calls.
A new 30 year pool at today's Freddie Mac rate, cash flows on Hull-White paths, and an
S-curve refinancing rule:

    CPR = 6% turnover (ramped over 30 months) + 45% / (1 + exp(-(WAC - m - 1%) / 0.2%))

where m is the mortgage rate on the path: the 10 year zero yield plus today's spread. The
pool is priced at par to fix its spread (no free MBS quotes), so the results are price
CHANGES and the option's value, which barely depend on that choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import hull_white as hw
import style as s
from curve import TreasuryCurve
from data import fred

TURNOVER, REFI_MAX, REFI_MID, REFI_WIDTH = 0.06, 0.45, 0.010, 0.002


def cash_flows(m, wac, coupon, refi=True):
    """Monthly cash flows and principal of a new 30 year pool along each mortgage rate path
    m (N x 361). Both returned as N x 360."""
    N = m.shape[0]
    bal = np.full(N, 100.0)
    w = wac / 12
    cf = np.zeros((N, 360)); principal = np.zeros((N, 360))
    for k in range(360):
        sched = bal * w / (1 - (1 + w) ** -(360 - k)) - bal * w
        cpr = TURNOVER * min(1.0, (k + 1) / 30)
        if refi:
            cpr = cpr + REFI_MAX / (1 + np.exp(-((wac - m[:, k]) - REFI_MID) / REFI_WIDTH))
        prep = (1 - (1 - cpr) ** (1 / 12)) * (bal - sched)
        cf[:, k] = bal * coupon / 12 + sched + prep
        principal[:, k] = sched + prep
        bal = bal - sched - prep
    return cf, principal


def run(curve: TreasuryCurve):
    mortgage = fred("MORTGAGE30US", "1990-01-01").loc[:curve.date]
    m0 = float(mortgage.iloc[-1]) / 100
    wac = m0
    coupon = round((wac - 0.005) * 400) / 400               # investor coupon, nearest quarter point
    # Mean reversion from the T-bill fit, but sigma set so the 10 year yield moves as much as
    # it did over the last year. With the T-bill sigma the 10 year moves ~40 bp a year, half
    # what it really does, and the refinancing option comes out near worthless. The mortgage
    # rate follows the 10 year, so that is the vol that matters here.
    a, *_ = hw.calibrate(curve)
    y10 = fred("DGS10", "2025-06-01").loc[:curve.date].iloc[-253:] / 100
    vol10 = float(y10.diff().dropna().std() * np.sqrt(252))
    sigma = vol10 * 10 * a / (1 - np.exp(-10 * a))          # sigma B(10) / 10 = vol10
    t, x = hw.simulate_x(a, sigma, 4000, 30, 12, seed=11, antithetic=True)
    dt = 1 / 12
    spread = m0 - float(curve.zero(10.0))                     # mortgage rate over the 10 year zero
    tk = t[1:]

    def scenario(c, x_):
        _, al = hw.alpha(c, a, sigma, t)
        r = x_ + al
        m = hw.zero_yield(c, a, sigma, t, r) + spread
        disc = np.exp(-np.cumsum((r[:, :-1] + r[:, 1:]) / 2 * dt, axis=1))   # to each month end
        return m, disc

    def price(cf, disc, oas):
        return float((cf * disc * np.exp(-oas * tk)).sum(1).mean())

    def average_life(principal):
        return float(((principal * tk).sum(1) / principal.sum(1)).mean())

    m, disc = scenario(curve, x)
    cf, _ = cash_flows(m, wac, coupon)
    oas = brentq(lambda o: price(cf, disc, o) - 100, -0.05, 0.10)
    m0_, d0_ = scenario(curve, np.zeros((1, x.shape[1])))      # zero vol: the forwards themselves
    cf0, _ = cash_flows(m0_, wac, coupon)
    zvol = brentq(lambda o: price(cf0, d0_, o) - 100, -0.05, 0.10)

    # whole curve shifted, pool and spread held: price with and without refinancing, average life
    shifts = np.linspace(-0.025, 0.025, 21)
    p_mbs, p_bond, life = [], [], []
    for sh in shifts:
        m2, d2 = scenario(TreasuryCurve(sh), x)
        cf2, pr2 = cash_flows(m2, wac, coupon)
        cfn, _ = cash_flows(m2, wac, coupon, refi=False)
        p_mbs.append(price(cf2, d2, oas)); p_bond.append(price(cfn, d2, oas)); life.append(average_life(pr2))
    p_mbs, p_bond, life = map(np.array, (p_mbs, p_bond, life))
    xm = (m0 + shifts) * 100
    i0 = len(shifts) // 2

    fig, ax = s.figure(8.4, 3.2)
    ax.plot(xm, p_bond, color=s.INK, lw=1.6, ls=(0, (4, 3)), label="same loans, nobody refinances")
    ax.plot(xm, p_mbs, color=s.AMBER, lw=2.8, label="MBS pool, homeowners can refinance")
    ax.fill_between(xm, p_mbs, p_bond, where=p_bond > p_mbs, color=s.AMBER, alpha=0.16, lw=0)
    ax.scatter([xm[i0]], [p_mbs[i0]], s=40, color=s.AMBER, zorder=4, edgecolor=s.DARK)
    ax.annotate("the homeowners' call", (xm[3], (p_mbs[3] + p_bond[3]) / 2), xytext=(18, 0),
                textcoords="offset points", color=s.AMBER, fontsize=8.6, fontweight="bold", va="center")
    ax.invert_xaxis()
    s.labels(ax, "30 year mortgage rate, whole curve shifted (%)", "price per 100",
             f"New {coupon * 100:.2f}% pool at today's {m0 * 100:.2f}% mortgage rate, {x.shape[0]:,} Hull-White paths")
    s.legend(ax, loc="upper left")
    s.save(fig, "mbs_price.png")

    fig, ax = s.figure(6.2, 4.0)
    ax.plot(mortgage.index, mortgage.values, color=s.INK, lw=1.1)
    ax.fill_between(mortgage.index, mortgage.values, m0 * 100, where=mortgage.values < m0 * 100, color="#6e7681",
                    alpha=0.45, lw=0, label="locked in: refinancing costs more")
    ax.fill_between(mortgage.index, mortgage.values, m0 * 100, where=mortgage.values > m0 * 100 + 1, color=s.GREEN,
                    alpha=0.35, lw=0, label="would refinance today")
    ax.axhline(m0 * 100, color=s.AMBER, lw=1.4, ls=(0, (4, 3)))
    ax.text(pd.Timestamp("2009-06-01"), m0 * 100 + 0.22, f"today {m0 * 100:.2f}%", color=s.AMBER, fontsize=8.4,
            fontweight="bold")
    s.labels(ax, None, "30 year mortgage rate (%)", "Freddie Mac 30 year rate since 1990")
    s.legend(ax, loc="upper right")
    s.save(fig, "mbs_lockin.png")

    fig, ax = s.figure(6.2, 4.0, light=True)
    ax.plot(xm, life, color=s.INDIGO, lw=2.6)
    ax.scatter([xm[i0]], [life[i0]], s=40, color=s.INDIGO, zorder=4)
    ax.annotate(f"today {life[i0]:.1f} years", (xm[i0], life[i0]), xytext=(10, -14), textcoords="offset points",
                color=s.LIGHT_INK, fontsize=8.4)
    ax.annotate(f"rates up 2.5 pts: {life[-1]:.1f} years", (xm[-1], life[-1]), xytext=(0, -16),
                textcoords="offset points", ha="right", va="top", color=s.LIGHT_INK, fontsize=8.4, fontweight="bold")
    ax.annotate(f"rates down 2.5 pts:\n{life[0]:.1f} years", (xm[0], life[0]), xytext=(0, 64),
                textcoords="offset points", ha="left", color=s.LIGHT_INK, fontsize=8.4, fontweight="bold")
    s.labels(ax, "30 year mortgage rate, whole curve shifted (%)", "average life (years)",
             "Rates up, the pool gets longer", light=True)
    s.save(fig, "mbs_life.png", light=True)

    lock = mortgage.loc["2020-03-01":"2021-12-31"]
    facts = dict(rate_date=str(mortgage.index[-1].date()), mortgage_rate=m0 * 100, coupon=coupon * 100,
                 vol10_bp=vol10 * 1e4, sigma=sigma * 100, a=a, paths=int(x.shape[0]),
                 oas_at_par_bp=oas * 1e4, zero_vol_spread_bp=zvol * 1e4, option_cost_bp=(zvol - oas) * 1e4,
                 option_value=float(p_bond[i0] - p_mbs[i0]),
                 down1_mbs=float(p_mbs[i0 - 4] - p_mbs[i0]), up1_mbs=float(p_mbs[i0] - p_mbs[i0 + 4]),
                 down1_no_refi=float(p_bond[i0 - 4] - p_bond[i0]), up1_no_refi=float(p_bond[i0] - p_bond[i0 + 4]),
                 life=float(life[i0]), life_up=float(life[-1]), life_down=float(life[0]),
                 rate_2020_2021_min=float(lock.min()), rate_2020_2021_max=float(lock.max()))
    s.record("mbs", facts)
    return facts
