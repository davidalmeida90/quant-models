"""Pricing the option inside three real bonds, with free data (as of 14 Oct 2026).

  DoorDash 0% convertible due 15 May 2030   (CUSIP 25809KAB1)  convertible = straight bond + call on the stock
  Viking Cruises 9.125% due 15 Jul 2031     (CUSIP 92676XAG2)  callable    = straight bond - call
  Strategy 0% convertible due 1 Dec 2029    (CUSIP 594972AS0)  putable     = straight bond + put

Sources: bond terms from the DoorDash 10-K/A, the Viking indenture (Section 3.07) and the Strategy 10-K;
trade prices from FINRA (last trade 14 Oct 2026); Treasury par yields from home.treasury.gov (14 Oct 2026);
stock prices and option quotes from Yahoo Finance.

    py -3 model.py
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TODAY = date(2026, 10, 15)                      # settlement of the 14 Oct trades


def yf(d):                                      # year fraction, act/365
    return (d - TODAY).days / 365.0


# ------------------------------------------------------------------ Treasury curve, 14 Oct 2026 (par yields, %)
CMT = {1 / 12: 4.11, 0.125: 4.15, 2 / 12: 4.15, 0.25: 4.28, 4 / 12: 4.36, 0.5: 4.37,
       1: 4.53, 2: 4.82, 3: 4.92, 5: 5.00, 7: 5.09, 10: 5.19, 20: 5.57, 30: 5.52}


def build_discount():
    """Bootstrap discount factors on a half-year grid from par yields (semiannual coupons)."""
    ten, yld = np.array(list(CMT)), np.array(list(CMT.values())) / 100
    grid = np.arange(0.5, 10.01, 0.5)
    par = np.interp(grid, ten, yld)
    dfs = []
    for n, y in enumerate(par, start=1):
        dfs.append((1 - (y / 2) * sum(dfs)) / (1 + y / 2))
    bills = np.array([t for t in CMT if t < 0.5])
    bill_df = [1 / (1 + CMT[t] / 100 * t) for t in bills]      # money-market style for bills
    t_all = np.concatenate([[0.0], bills, grid])
    logdf = np.log(np.concatenate([[1.0], bill_df, dfs]))
    return lambda t: float(np.exp(np.interp(t, t_all, logdf)))


DF = build_discount()


def zero(t):                                    # continuously compounded zero rate
    return -math.log(DF(t)) / t


# ------------------------------------------------------------------ 1) Yield to worst (Viking)
def bond_yield(clean, cpn, settle, end, redeem, last_cpn):
    """Street yield, semiannual, 30/360, to a given redemption date and price."""
    def d360(a, b):
        return 360 * (b.year - a.year) + 30 * (b.month - a.month) + (min(b.day, 30) - min(a.day, 30))
    dirty = clean + cpn * d360(last_cpn, settle) / 360
    dates, d = [], end
    while d > settle:
        dates.append(d)
        d = date(d.year - (d.month <= 6), (d.month - 7) % 12 + 1, d.day)
    def pv(y):
        return sum((cpn / 2) / (1 + y / 2) ** (d360(settle, x) / 180) for x in dates) + \
            redeem / (1 + y / 2) ** (d360(settle, end) / 180) - dirty
    return brentq(pv, -0.5, 1.0)


# ------------------------------------------------------------------ 2) Convertible: bond + Black Scholes
def bs_call(S, K, T, r, vol):
    d1 = (math.log(S / K) + (r + vol ** 2 / 2) * T) / (vol * math.sqrt(T))
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - vol * math.sqrt(T))


# ------------------------------------------------------------------ 3) Convertible: Tsiveriotis-Fernandes tree
def tf_tree(S0, ratio, T, r, vol, spread, call_from=None, call_trigger=None, call_price=100.0, n=1000,
            put_at=None, put_price=100.0):
    """Binomial stock tree. Per 100 of face: `ratio` shares. Equity part discounted at r, debt part at r + spread.
    Issuer soft call from `call_from` (years) when the stock is at or above `call_trigger`.
    Holder put on the single date `put_at` (years): the bond is worth at least `put_price` there."""
    put_step = None if put_at is None else round(put_at / (T / n))
    dt = T / n
    u = math.exp(vol * math.sqrt(dt)); d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    dr, drs = math.exp(-r * dt), math.exp(-(r + spread) * dt)
    S = S0 * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)
    conv = ratio * S
    E = np.where(conv >= 100, conv, 0.0)
    D = np.where(conv >= 100, 0.0, 100.0)
    for i in range(n - 1, -1, -1):
        S = S0 * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        conv = ratio * S
        E = dr * (p * E[:-1] + (1 - p) * E[1:])
        D = drs * (p * D[:-1] + (1 - p) * D[1:])
        hold = E + D
        called = np.zeros_like(hold, dtype=bool)
        if call_from is not None and i * dt >= call_from:
            called = (S >= call_trigger) & (hold > call_price)       # issuer calls when holding is worth more than the call price
        convert = (conv >= hold) | (called & (conv >= call_price))   # holder converts if it beats holding, or beats the call price
        redeem = called & ~convert
        E = np.where(convert, conv, np.where(redeem, 0.0, E))
        D = np.where(convert, 0.0, np.where(redeem, call_price, D))
        if put_step is not None and i == put_step:
            put = (E + D) < put_price                                 # holder sells the bond back
            E = np.where(put, 0.0, E)
            D = np.where(put, put_price, D)
        if i == 1:
            v1, s1 = E + D, S
    return float(E[0] + D[0]), float((v1[0] - v1[1]) / (s1[0] - s1[1]))


# ------------------------------------------------------------------ 4) Interest rate tree (lognormal, Black Derman Toy style)
def rate_tree(times, vol):
    """Short rates r[i][j] on the steps between times[i] and times[i+1], calibrated to the Treasury curve."""
    rates, Q = [], np.array([1.0])
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        spacing = np.exp(2 * vol * math.sqrt(dt) * np.arange(i + 1))
        target = DF(times[i + 1])
        low = brentq(lambda r0: float(np.sum(Q * np.exp(-r0 * spacing * dt))) - target, -0.05, 0.5)
        r = low * spacing
        rates.append(r)
        disc = Q * np.exp(-r * dt) / 2
        Q = np.concatenate([[0.0], disc]) + np.concatenate([disc, [0.0]])
    return rates


def tree_price(times, rates, flows, oas, rule=None):
    """Backward walk. flows[i] = cash paid at times[i]; rule(i, v) applies the call (min) or put (max) after the flow."""
    v = np.full(len(rates[-1]) + 1, flows[-1])
    for i in range(len(times) - 2, -1, -1):
        dt = times[i + 1] - times[i]
        r = rates[i] + oas
        ex = np.exp(-r * dt) * (v[:-1] + v[1:]) / 2            # value just after the cash flow at times[i]
        if rule is not None and i > 0:
            ex = rule(i, ex)
        v = ex + (flows[i] if i > 0 else 0.0)
    return float(v[0])


def main():
    print(f"Treasury zero rates: 1y {zero(1):.2%}  1.6y {zero(1.6):.2%}  3.6y {zero(3.58):.2%}  4.75y {zero(4.75):.2%}")

    # ---------------- Viking: yield to worst
    v_px, v_cpn, last = 104.50, 9.125, date(2026, 7, 15)
    print("\nVIKING 9.125% 2031, last trade 104.50")
    for lbl, end, red in [("Jul 2027 @102.281", date(2027, 7, 15), 102.281), ("Jul 2028 @100", date(2028, 7, 15), 100.0),
                          ("maturity 2031", date(2031, 7, 15), 100.0)]:
        print(f"  yield to {lbl:18s} {bond_yield(v_px, v_cpn, TODAY, end, red, last):.3%}")

    # ---------------- Viking: rate tree, callable vs straight
    RATE_VOL = 0.20
    cdates = [date(2027 + k // 2, 1 if k % 2 == 0 else 7, 15) for k in range(10)]
    times = [0.0] + [yf(x) for x in cdates]
    flows = [0.0] + [v_cpn / 2] * 9 + [100 + v_cpn / 2]
    callpx = {1: 104.563, 2: 102.281, 3: 102.281}                 # Jan 27, Jul 27, Jan 28; par from Jul 2028
    rates = rate_tree(times, RATE_VOL)
    accrued = v_cpn / 2 * 0.5                                    # Jul 15 to Oct 15, 30/360
    callable_rule = lambda i, ex: np.minimum(ex, callpx.get(i, 100.0))
    oas = brentq(lambda s: tree_price(times, rates, flows, s, callable_rule) - accrued - v_px, -0.05, 0.3)
    straight = tree_price(times, rates, flows, oas) - accrued
    print(f"  OAS {oas * 1e4:.0f} bp | straight {straight:.2f} | callable {v_px:.2f} | call worth {straight - v_px:.2f} points")

    # ---------------- Strategy: same stock tree as DoorDash plus one rule, the holder put on 1 Jun 2028.
    # Stock tree because this bond also converts (1.4872 shares per 1,000) and the put and the conversion interact.
    s_px, s_S, s_ratio, s_K = 90.43, 143.24, 0.14872, 672.40
    t_put, t_mat, s_vol = yf(date(2028, 6, 1)), yf(date(2029, 12, 1)), 0.71   # vol: Yahoo, Dec 2028 / Jan 2029 calls
    s_r = zero(t_mat)
    s_kw = dict(call_from=yf(date(2026, 12, 4)), call_trigger=1.3 * s_K, n=800)
    full = lambda spr, put=True, ratio=s_ratio: tf_tree(s_S, ratio, t_mat, s_r, s_vol, spr,
                                                       put_at=t_put if put else None, **s_kw)[0]
    s_spread = brentq(lambda spr: full(spr) - s_px, -0.02, 0.3)
    no_put, no_conv = full(s_spread, put=False), full(s_spread, ratio=1e-9)
    straight_s = 100 * math.exp(-(s_r + s_spread) * t_mat)
    print(f"\nSTRATEGY 0% 2029, last trade 90.43, put 1 Jun 2028 @100, stock 143.24, vol {s_vol:.0%}")
    print(f"  yield to put {(100 / s_px) ** (1 / t_put) - 1:.2%} | yield to maturity {(100 / s_px) ** (1 / t_mat) - 1:.2%} | Treasury 1.6y {zero(t_put):.2%} (cont.)")
    print(f"  implied credit spread {s_spread * 1e4:.0f} bp | straight bond {straight_s:.2f}")
    print(f"  without the put {no_put:.2f} -> put worth {s_px - no_put:.2f} | without conversion {no_conv:.2f} -> conversion worth {s_px - no_conv:.2f}")
    print(f"  conversion value today {s_ratio * s_S:.1f}")

    # ---------------- DoorDash: tree calibrated to the market, then Black Scholes with the same inputs
    S0, ratio, K, d_px = 184.01, 0.3425, 291.97, 94.48
    T, VOL = yf(date(2030, 5, 15)), 0.48
    r = zero(T)
    tc, trig = yf(date(2028, 5, 20)), 1.3 * K
    spread = brentq(lambda s: tf_tree(S0, ratio, T, r, VOL, s, tc, trig)[0] - d_px, -0.02, 0.2)
    tree, delta = tf_tree(S0, ratio, T, r, VOL, spread, tc, trig)
    floor = 100 * math.exp(-(r + spread) * T)
    call = bs_call(S0, K, T, r, VOL)
    print(f"\nDOORDASH 0% 2030, last trade 94.48, stock 184.01, vol {VOL:.0%}, Treasury {r:.2%}")
    print(f"  conversion value {ratio * S0:.1f} | premium {d_px / (ratio * S0) - 1:.0%}")
    print(f"  implied credit spread {spread * 1e4:.0f} bp -> bond floor {floor:.2f}")
    print(f"  Black Scholes: one call {call:.2f} $ -> x3.425 = {ratio * call:.2f} points -> bond + call = {floor + ratio * call:.2f}")
    print(f"  Tree (TF, soft call May 2028 at 130%): {tree:.2f} | delta {delta:.4f} per 100 face = {delta * 10:.2f} shares per bond "
          f"({delta / ratio:.0%} of the 3.425 conversion shares)")
    for s_alt in (0.01, 0.02):
        t_alt, _ = tf_tree(S0, ratio, T, r, VOL, s_alt, tc, trig)
        print(f"    check: spread {s_alt * 1e4:.0f} bp -> tree {t_alt:.2f}")


if __name__ == "__main__":
    main()
