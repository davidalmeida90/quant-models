"""Vega neutral volga spread. The standalone script behind the write-up at
davidariasfinance.com/scripts/volga-convexity-spread/

Long strangle, short straddle, wings sized so total vega is exactly zero. What is
left on the book is the second derivative of value against volatility.

    py -3 volga_spread_page.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

S, R, Q = 100.0, 0.05, 0.0
TAU = 60.0 / 365.0

# Smile fitted to a real SPY chain, quadratic in log moneyness.
# iv(m) = 0.1544 - 0.3159 m + 1.2915 m^2, residual 1.2 vol points over 163 quotes.
A1, A2 = -0.3159, 1.2915


def leg_vol(K, atm):
    """Each strike carries its own implied vol. The put wing is steep and monotone,
    the call wing dips then curls back up, which is what the market quotes."""
    m = np.log(np.asarray(K, dtype=float) / S)
    return np.maximum(atm + A1 * m + A2 * m * m, 0.05)


def _d(K, sig, tau=TAU):
    d1 = (np.log(S / K) + (R - Q + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
    return d1, d1 - sig * np.sqrt(tau)


def call(K, sig, tau=TAU):
    d1, d2 = _d(K, sig, tau)
    return S * np.exp(-Q * tau) * norm.cdf(d1) - K * np.exp(-R * tau) * norm.cdf(d2)


def put(K, sig, tau=TAU):
    d1, d2 = _d(K, sig, tau)
    return K * np.exp(-R * tau) * norm.cdf(-d2) - S * np.exp(-Q * tau) * norm.cdf(-d1)


def vega(K, sig, tau=TAU):
    d1, _ = _d(K, sig, tau)
    return S * np.exp(-Q * tau) * norm.pdf(d1) * np.sqrt(tau)


def volga(K, sig, tau=TAU):
    """Second derivative of value against vol. Vega times d1 d2 over sigma."""
    d1, d2 = _d(K, sig, tau)
    return vega(K, sig, tau) * d1 * d2 / sig


def gamma(K, sig, tau=TAU):
    d1, _ = _d(K, sig, tau)
    return np.exp(-Q * tau) * norm.pdf(d1) / (S * sig * np.sqrt(tau))


def theta_call(K, sig, tau=TAU):
    """Per year. Negative means the option bleeds."""
    d1, d2 = _d(K, sig, tau)
    return (-S * norm.pdf(d1) * sig / (2.0 * np.sqrt(tau))
            - R * K * np.exp(-R * tau) * norm.cdf(d2))


def theta_put(K, sig, tau=TAU):
    d1, d2 = _d(K, sig, tau)
    return (-S * norm.pdf(d1) * sig / (2.0 * np.sqrt(tau))
            + R * K * np.exp(-R * tau) * norm.cdf(-d2))


def spread(half, atm=0.18):
    """Long strangle at S +/- half, short straddle at S, wings scaled so net vega
    is zero. Returns every number the write-up quotes."""
    kl, kh = S - half, S + half
    sl, sh = float(leg_vol(kl, atm)), float(leg_vol(kh, atm))

    v_std = 2.0 * vega(S, atm)
    o_std = 2.0 * volga(S, atm)
    g_std = 2.0 * gamma(S, atm)
    c_std = call(S, atm) + put(S, atm)

    v_stg = vega(kh, sh) + vega(kl, sl)        # ONE strangle
    o_stg = volga(kh, sh) + volga(kl, sl)
    g_stg = gamma(kh, sh) + gamma(kl, sl)
    c_stg = call(kh, sh) + put(kl, sl)
    t_std = theta_call(S, atm) + theta_put(S, atm)
    t_stg = theta_call(kh, sh) + theta_put(kl, sl)   # the actual two wing legs

    lam = v_std / v_stg                         # contracts that flatten vega
    return dict(kl=kl, kh=kh, iv_put=sl, iv_call=sh, lam=lam,
                vega_1=v_stg, vega_std=v_std,
                vega_net_1x1=v_stg - v_std, volga_net_1x1=o_stg - o_std,
                vega_net=lam * v_stg - v_std,
                volga_net=lam * o_stg - o_std,
                gamma_net=lam * g_stg - g_std,
                cost_std=c_std, cost_stg=lam * c_stg,
                credit=c_std - lam * c_stg,
                theta_net=(lam * t_stg - t_std) / 365.0)


def repriced(half, dsig, atm=0.18):
    """Actual P&L of the structure after a parallel shift in implied vol, priced
    leg by leg. No approximation."""
    b = spread(half, atm)
    kl, kh, lam = b["kl"], b["kh"], b["lam"]

    def val(bump):
        stg = lam * (call(kh, b["iv_call"] + bump) + put(kl, b["iv_put"] + bump))
        std = call(S, atm + bump) + put(S, atm + bump)
        return stg - std

    return val(dsig) - val(0.0)


if __name__ == "__main__":
    WIDTHS = [5, 8, 11, 14, 18, 22, 28, 36, 45]

    print("=== one for one is short vega ===")
    print(f"{'strikes':>12}{'vega 1 stg':>12}{'net vega 1:1':>14}"
          f"{'net volga 1:1':>15}{'strangles for 0':>17}")
    for h in WIDTHS:
        b = spread(h)
        print(f"{b['kl']:5.0f}/{b['kh']:<6.0f}{b['vega_1']:12.2f}"
              f"{b['vega_net_1x1']:14.2f}{b['volga_net_1x1']:15.1f}{b['lam']:17.2f}")

    print("\n=== vega matched: what you actually own ===")
    print(f"{'strikes':>12}{'lambda':>9}{'vega':>8}{'volga':>9}{'gamma':>10}"
          f"{'put IV':>9}{'call IV':>9}{'credit':>9}{'theta/day':>11}")
    for h in WIDTHS:
        b = spread(h)
        print(f"{b['kl']:5.0f}/{b['kh']:<6.0f}{b['lam']:9.2f}{b['vega_net']:8.2f}"
              f"{b['volga_net']:9.0f}{b['gamma_net']:10.4f}"
              f"{b['iv_put']:8.1%}{b['iv_call']:9.1%}"
              f"{b['credit']:9.2f}{b['theta_net']:11.3f}")

    print("\n=== the quadratic vs the real reprice, at the peak ===")
    hstar = max(WIDTHS, key=lambda h: spread(h)["volga_net"])
    b = spread(hstar)
    print(f"peak at {b['kl']:.0f}/{b['kh']:.0f}, volga {b['volga_net']:.0f}")
    print(f"{'vol move':>10}{'0.5 x volga x dsig^2':>22}{'repriced':>12}{'gap':>8}")
    for dv in (0.02, 0.05, 0.10, 0.15):
        approx = 0.5 * b["volga_net"] * dv * dv
        real = repriced(hstar, dv)
        print(f"{dv:+10.0%}{approx:22.2f}{real:12.2f}{real - approx:8.2f}")

    print("\n=== vol regime: where the convexity lives ===")
    REGIMES = [0.12, 0.15, 0.18, 0.22, 0.28, 0.35]
    print(f"{'ATM vol':>9}" + "".join(f"{f'{S-h:.0f}/{S+h:.0f}':>11}" for h in WIDTHS))
    for atm in REGIMES:
        row = "".join(f"{spread(h, atm)['volga_net']:11.0f}" for h in WIDTHS)
        print(f"{atm:9.0%}{row}")
    print("\npeak width by regime")
    grid = np.arange(3.0, 45.1, 0.5)
    for atm in REGIMES:
        vals = [spread(float(h), atm)["volga_net"] for h in grid]
        k = int(np.argmax(vals))
        print(f"  ATM {atm:.0%}  peaks at {S-grid[k]:.0f}/{S+grid[k]:.0f}"
              f"  volga {vals[k]:.0f}  lambda {spread(float(grid[k]), atm)['lam']:.2f}")
