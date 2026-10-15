"""1. Callable and puttable bonds on a Black-Derman-Toy tree.

The tree is calibrated to today's Treasury curve by forward induction on Arrow-Debreu prices,
with a constant rate vol measured from one year of daily log changes in the 1 year yield.
A 10 year bond, callable (issuer's option) or puttable (holder's option) at par on every
coupon date from year 2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import style as s
from curve import TreasuryCurve
from data import fred


def bdt(curve: TreasuryCurve, sigma: float, n: int, dt: float) -> np.ndarray:
    """r(i, j) = a_i exp(2 sigma sqrt(dt) j), each a_i fitted so the tree reprices the
    discount factor of the next date."""
    R = np.zeros((n, n))
    Q = np.array([1.0])                                  # Arrow-Debreu prices of today's nodes
    for i in range(n):
        j = np.arange(i + 1)
        target = float(curve.df((i + 1) * dt))
        a = brentq(lambda a: float((Q * np.exp(-a * np.exp(2 * sigma * np.sqrt(dt) * j) * dt)).sum()) - target,
                   1e-6, 1.0)
        r = a * np.exp(2 * sigma * np.sqrt(dt) * j)
        R[i, :i + 1] = r
        d = Q * np.exp(-r * dt)
        Qn = np.zeros(i + 2)
        Qn[:-1] += 0.5 * d
        Qn[1:] += 0.5 * d
        Q = Qn
    return R


def bond_price(R, dt, T, coupon, kind="straight", K=100.0, first=2.0, freq=2, mark=False):
    """Backward induction. kind "call": the issuer calls at K on any coupon date from `first`;
    kind "put": the holder puts at K on the same dates."""
    n = int(round(T / dt))
    step = int(round(1 / (freq * dt)))
    c = 100 * coupon / freq
    V = np.full(n + 1, 100.0 + c)
    exercised = {}
    for i in range(n - 1, -1, -1):
        r = R[i, :i + 1]
        cont = np.exp(-r * dt) * 0.5 * (V[1:i + 2] + V[:i + 1])
        if i > 0 and i % step == 0:
            if i * dt >= first - 1e-9:
                if kind == "call":
                    exercised[i] = cont > K
                    cont = np.minimum(cont, K)
                elif kind == "put":
                    exercised[i] = cont < K
                    cont = np.maximum(cont, K)
            cont = cont + c
        V = cont
    return (float(V[0]), exercised) if mark else float(V[0])


def run():
    curve = TreasuryCurve()
    y1 = fred("DGS1", "2024-06-01").loc[:curve.date].iloc[-260:]
    sigma = float(np.log(y1).diff().dropna().std() * np.sqrt(252))
    y10 = curve.par["DGS10"]
    coupon = round(y10 * 400) / 400                       # nearest quarter point
    T, dt = 10.0, 0.5
    n = int(T / dt)
    R = bdt(curve, sigma, n, dt)
    straight = bond_price(R, dt, T, coupon)
    call, ex_call = bond_price(R, dt, T, coupon, "call", mark=True)
    put, ex_put = bond_price(R, dt, T, coupon, "put", mark=True)

    # price against a parallel shift of the whole curve, the tree recalibrated each time
    shifts = np.linspace(-0.025, 0.025, 41)
    lines = {"straight": [], "call": [], "put": []}
    for sh in shifts:
        R2 = bdt(TreasuryCurve(sh), sigma, n, dt)
        for k in lines:
            lines[k].append(bond_price(R2, dt, T, coupon, k))
    x = (y10 + shifts) * 100

    # option value against rate vol
    sigmas = np.linspace(0.04, 0.40, 19)
    v_call, v_put = [], []
    for sg in sigmas:
        R3 = bdt(curve, sg, n, dt)
        st = bond_price(R3, dt, T, coupon)
        v_call.append(st - bond_price(R3, dt, T, coupon, "call"))
        v_put.append(bond_price(R3, dt, T, coupon, "put") - st)

    fig, ax = s.figure(6.2, 4.0, light=True)
    ax.plot(x, lines["straight"], color=s.LIGHT_INK, lw=2.0, label="straight bond")
    ax.plot(x, lines["call"], color=s.ORANGE, lw=2.4, label="callable at par from year 2")
    ax.plot(x, lines["put"], color=s.INDIGO, lw=2.4, label="puttable at par from year 2")
    ax.axhline(100, color=s.LIGHT_GRID, lw=1.0, ls=(0, (3, 3)))
    ax.axvline(y10 * 100, color=s.LIGHT_INK, lw=0.8, ls=(0, (2, 3)), alpha=0.6)
    ax.text(y10 * 100 + 0.06, max(lines["straight"]) * 0.995, f"today {y10 * 100:.2f}%",
            color=s.LIGHT_INK, fontsize=8.2, va="top")
    s.labels(ax, "10 year Treasury yield, whole curve shifted (%)", "price per 100",
             f"{T:.0f} year {coupon * 100:.2f}% bond on today's Treasury tree", light=True)
    s.legend(ax, light=True, loc="upper right")
    s.save(fig, "callable_price.png", light=True)

    fig, ax = s.figure(6.2, 4.0)
    ax.plot(sigmas * 100, v_call, color=s.AMBER, lw=2.4, label="call the issuer owns")
    ax.plot(sigmas * 100, v_put, color=s.CYAN, lw=2.4, label="put the holder owns")
    ax.axvline(sigma * 100, color=s.INK, lw=0.8, ls=(0, (2, 3)), alpha=0.7)
    ax.text(sigma * 100 + 0.6, max(v_call) * 0.97, f"measured vol {sigma * 100:.0f}%", color=s.INK,
            fontsize=8.2, va="top")
    s.labels(ax, "short rate volatility in the tree (%)", "option value per 100", "Both options gain value with rate vol")
    s.legend(ax, loc="upper left")
    s.save(fig, "callable_vol.png")

    fig, ax = s.figure(6.2, 4.0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    m = 10                                                # first five years
    for i in range(m - 1):
        for j in range(i + 1):
            for jj in (j, j + 1):
                ax.plot([i * dt, (i + 1) * dt], [R[i, j] * 100, R[i + 1, jj] * 100], color=s.INK, lw=0.5, alpha=0.28)
    for i in range(m):
        colors = np.array([s.INK] * (i + 1), dtype=object)
        if i in ex_call:
            colors[ex_call[i]] = s.AMBER
        if i in ex_put:
            colors[ex_put[i]] = s.CYAN
        ax.scatter(np.full(i + 1, i * dt), R[i, :i + 1] * 100, s=26, c=list(colors), zorder=3, edgecolor=s.DARK, lw=0.6)
    for c_, lab in ((s.AMBER, "issuer calls the bond"), (s.CYAN, "holder puts the bond"), (s.INK, "nobody exercises")):
        ax.scatter([], [], s=26, c=c_, label=lab)
    s.labels(ax, "years", "6 month rate (%)", f"BDT tree, vol {sigma * 100:.0f}%, first {m * dt:.0f} years")
    s.legend(ax, loc="upper left")
    s.save(fig, "bdt_tree.png")

    facts = dict(date=str(curve.date.date()), y10=y10 * 100, sigma=sigma * 100, coupon=coupon * 100,
                 straight=straight, callable=call, puttable=put, call_value=straight - call,
                 put_value=put - straight, steps=n, nss_fit_error_bp=curve.fit_error_bp)
    s.record("callable_puttable", facts)
    return curve, facts
