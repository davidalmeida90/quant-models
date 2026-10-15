"""5. Hull-White one factor: dr = [theta(t) - a r] dt + sigma dW.

a and sigma come from an AR(1) on month end 3 month T-bill yields since 1990; theta(t) is
fitted so today's Treasury curve comes out exactly. Paths are simulated exactly as
r(t) = x(t) + alpha(t), with x a zero mean Ornstein-Uhlenbeck process. The same engine
drives the MBS pool's rate paths.
"""
from __future__ import annotations

import numpy as np

import style as s
from curve import TreasuryCurve
from data import fred


def calibrate(curve: TreasuryCurve):
    tb = fred("DTB3", "1990-01-01").loc[:curve.date] / 100
    monthly = tb.resample("ME").last().dropna()
    x0, x1 = monthly.values[:-1], monthly.values[1:]
    phi, c0 = np.polyfit(x0, x1, 1)
    dt = 1 / 12
    a = float(-np.log(phi) / dt)
    res = x1 - (c0 + phi * x0)
    sigma = float(res.std() * np.sqrt(2 * a / (1 - np.exp(-2 * a * dt))))
    return a, sigma, float(phi), tb, monthly


def simulate_x(a, sigma, N, years, per_year, seed=7, antithetic=False):
    """The zero mean part x(t), drawn exactly over each step."""
    rng = np.random.default_rng(seed)
    steps = years * per_year
    t = np.linspace(0, years, steps + 1)
    dt = t[1] - t[0]
    decay = np.exp(-a * dt)
    sd = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    z = rng.standard_normal((N // 2 if antithetic else N, steps))
    if antithetic:
        z = np.vstack([z, -z])
    x = np.zeros((z.shape[0], steps + 1))
    for i in range(steps):
        x[:, i + 1] = x[:, i] * decay + sd * z[:, i]
    return t, x


def alpha(curve: TreasuryCurve, a, sigma, t):
    """alpha(t) puts the mean of r on today's forward curve."""
    f0 = curve.fwd(np.maximum(t, 1e-4))
    return f0, f0 + sigma ** 2 / (2 * a ** 2) * (1 - np.exp(-a * t)) ** 2


def zero_yield(curve: TreasuryCurve, a, sigma, t, r, tau=10.0):
    """tau year zero yield at each node, in closed form: P(t, t+tau) = A exp(-B r)."""
    B = (1 - np.exp(-a * tau)) / a
    f0 = curve.fwd(np.maximum(t, 1e-4))
    lnA = (np.log(curve.df(t + tau)) - np.log(curve.df(t)) + B * f0
           - sigma ** 2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B ** 2)
    return -(lnA - B * r) / tau


def run(curve: TreasuryCurve):
    a, sigma, phi, tb, monthly = calibrate(curve)
    rng = np.random.default_rng(7)
    N, years, steps = 5000, 10, 120
    t = np.linspace(0, years, steps + 1)
    dt = t[1] - t[0]
    x = np.zeros((N, steps + 1))
    decay = np.exp(-a * dt)
    sd = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    for i in range(steps):
        x[:, i + 1] = x[:, i] * decay + sd * rng.standard_normal(N)
    f0, al = alpha(curve, a, sigma, t)
    r = x + al
    # fit check: Monte Carlo 5 year zero coupon price against the market curve
    i5 = int(5 / dt)
    integral = (r[:, :i5] + r[:, 1:i5 + 1]).sum(1) * dt / 2
    zcb_mc, zcb_mkt = float(np.exp(-integral).mean()), float(curve.df(5.0))
    r5 = r[:, i5] * 100

    fig, ax = s.figure(6.2, 4.0)
    q = np.percentile(r * 100, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(t, q[0], q[4], color=s.VIOLET, alpha=0.16, lw=0, label="5 to 95% of paths")
    ax.fill_between(t, q[1], q[3], color=s.VIOLET, alpha=0.30, lw=0, label="25 to 75%")
    for k in range(14):
        ax.plot(t, r[k] * 100, color=s.INK, lw=0.55, alpha=0.35)
    ax.plot(t, f0 * 100, color=s.AMBER, lw=2.0, label="today's forward curve")
    ax.axhline(0, color=s.GRID, lw=1.0)
    s.labels(ax, "years ahead", "short rate (%)", f"Hull-White, a = {a:.2f}, sigma = {sigma * 100:.2f}%")
    s.legend(ax, loc="upper left")
    s.save(fig, "hw_fan.png")

    fig, ax = s.figure(6.2, 4.0, light=True)
    bins = np.linspace(np.percentile(r5, 0.2), np.percentile(r5, 99.8), 60)
    p5, p95 = np.percentile(r5, [5, 95])
    inside = (r5 >= p5) & (r5 <= p95)
    ax.hist(r5[inside], bins=bins, color=s.GRAPE, alpha=0.85, label="90% of paths")
    ax.hist(r5[~inside], bins=bins, color="#adb5bd", alpha=0.9, label="outer 10%")
    f5 = float(f0[i5] * 100)
    ax.axvline(f5, color=s.ORANGE, lw=1.6, ls=(0, (4, 2)))
    ax.text(f5 + 0.08, ax.get_ylim()[1] * 0.97, f"today's forward {f5:.2f}%", color=s.ORANGE,
            fontsize=8.4, va="top", fontweight="bold")
    ax.text(0.02, 0.95, f"90% between {p5:.1f}% and {p95:.1f}%", transform=ax.transAxes, va="top",
            color=s.LIGHT_INK, fontsize=8.6, fontweight="bold")
    s.labels(ax, "3 month rate in 5 years (%)", "paths", f"{N:,} Hull-White paths, 5 years ahead", light=True)
    s.legend(ax, light=True, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    s.save(fig, "hw_5y.png", light=True)

    facts = dict(a=a, sigma=sigma * 100, phi=phi, r0=float(tb.iloc[-1] * 100), half_life=float(np.log(2) / a),
                 zcb_mc=zcb_mc, zcb_market=zcb_mkt, error_bp=(zcb_mc - zcb_mkt) * 1e4,
                 r5_median=float(np.median(r5)), r5_p5=float(p5), r5_p95=float(p95),
                 since=str(monthly.index[0].date()), paths=N)
    s.record("hull_white", facts)
    return facts
