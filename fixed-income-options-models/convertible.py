"""2. Convertible bonds, Tsiveriotis-Fernandes (1998) on a CRR stock tree.

Cash the bond still owes is discounted at Treasuries plus a credit spread; the part that ends
up as shares is discounted risk free. The bond is hypothetical: 5 years, 1% coupon, converting
30% above today's UBER price, with the ICE BofA BBB spread from FRED.
"""
from __future__ import annotations

import numpy as np

import style as s
from curve import TreasuryCurve
from data import fred, prices


def tsiveriotis_fernandes(S0, sigma, r, spread, T=5.0, coupon=0.01, freq=2, CR=None, n=500):
    """Returns (cash part, equity part); their sum is the convertible's value."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    step = int(round(n / (T * freq)))
    c = 100 * coupon / freq
    S = S0 * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)
    conv = CR * S
    B = np.where(conv > 100 + c, 0.0, 100.0 + c)
    E = np.where(conv > 100 + c, conv, 0.0)
    for i in range(n - 1, -1, -1):
        S = S[:-1] / u
        B = np.exp(-(r + spread) * dt) * (p * B[:-1] + (1 - p) * B[1:])   # cash owed carries credit risk
        E = np.exp(-r * dt) * (p * E[:-1] + (1 - p) * E[1:])              # shares carry none
        if i > 0 and i % step == 0:
            B = B + c
        conv = CR * S
        convert = conv > B + E
        E = np.where(convert, conv, E)
        B = np.where(convert, 0.0, B)
    return float(B[0]), float(E[0])


def run(curve: TreasuryCurve):
    tk = "UBER"
    px = prices(tk, "2024-01-01", adjusted=True).loc[:curve.date]
    S0 = float(px.iloc[-1])
    sigma = float(np.log(px).diff().dropna().iloc[-252:].std() * np.sqrt(252))
    spread = float(fred("BAMLC0A4CBBB", "2025-06-01").loc[:curve.date].iloc[-1]) / 100
    T, coupon, premium = 5.0, 0.01, 0.30
    r = float(curve.zero(T))
    CR = 100 / ((1 + premium) * S0)
    B, E = tsiveriotis_fernandes(S0, sigma, r, spread, T, coupon, CR=CR)
    floor = sum(100 * coupon / 2 * np.exp(-(r + spread) * k / 2) for k in range(1, int(T * 2) + 1)) + \
        100 * np.exp(-(r + spread) * T)
    parity = CR * S0

    Ss = np.linspace(0.25 * S0, 2.4 * S0, 46)
    vals, bs, es = [], [], []
    for x in Ss:
        b, e = tsiveriotis_fernandes(x, sigma, r, spread, T, coupon, CR=CR, n=1500)   # fine steps
        vals.append(b + e); bs.append(b); es.append(e)
    vals, bs, es = map(np.array, (vals, bs, es))

    fig, ax = s.figure(6.2, 4.0)
    ax.plot(Ss, CR * Ss, color=s.INK, lw=1.2, ls=(0, (4, 3)), alpha=0.8, label="conversion value")
    ax.axhline(floor, color=s.PINK, lw=1.4, ls=(0, (2, 2)), label="bond floor")
    ax.plot(Ss, vals, color=s.GREEN, lw=2.6, label="convertible, TF tree")
    ax.scatter([S0], [B + E], s=40, color=s.GREEN, zorder=4, edgecolor=s.DARK)
    ax.annotate(f"today {B + E:.1f}", (S0, B + E), xytext=(-8, 12), textcoords="offset points",
                color=s.INK, fontsize=8.3, ha="right")
    s.labels(ax, f"{tk} share price ($)", "value per 100 face",
             f"5 year {coupon * 100:.0f}% convertible on {tk}, converts {premium * 100:.0f}% above today")
    s.legend(ax, loc="upper left")
    ax.set_ylim(0, max(vals) * 1.05)
    s.save(fig, "convertible_value.png")

    fig, ax = s.figure(6.2, 4.0, light=True)
    tot = bs + es
    ax.stackplot(Ss, es / tot * 100, bs / tot * 100, colors=[s.TEAL, "#adb5bd"], alpha=0.85,
                 labels=["equity part, discounted at Treasuries", "cash part, discounted at Treasuries + spread"])
    ax.axvline(S0, color=s.LIGHT_INK, lw=0.9, ls=(0, (2, 3)))
    ax.text(S0 * 1.02, 95, "today", color=s.LIGHT_INK, fontsize=8.2, va="top")
    ax.set_ylim(0, 100); ax.set_xlim(Ss[0], Ss[-1])
    s.labels(ax, f"{tk} share price ($)", "share of the value (%)", "Tsiveriotis-Fernandes split", light=True)
    s.legend(ax, light=True, loc="lower right")
    s.save(fig, "convertible_split.png", light=True)

    facts = dict(ticker=tk, S0=S0, sigma=sigma * 100, r=r * 100, spread=spread * 100, T=T, coupon=coupon * 100,
                 premium=premium * 100, CR=CR, price=B + E, cash_part=B, equity_part=E, floor=float(floor),
                 parity=parity, option=float(B + E - floor), price_date=str(px.index[-1].date()))
    s.record("convertible", facts)
    return facts
