"""Treasury zero curve for one day: CMT par yields from FRED, bootstrapped, then smoothed with
Nelson-Siegel-Svensson (the functional form the Fed's own Treasury curve uses)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from data import fred

PAR = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3,
       "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}


def nss(t, b):
    t = np.maximum(np.asarray(t, float), 1e-6)
    b0, b1, b2, b3, t1, t2 = b
    e1, e2 = np.exp(-t / t1), np.exp(-t / t2)
    f1 = (1 - e1) / (t / t1)
    return b0 + b1 * f1 + b2 * (f1 - e1) + b3 * ((1 - e2) / (t / t2) - e2)


class TreasuryCurve:
    """`shift` moves every par yield in parallel (used for the price against rates charts)."""

    def __init__(self, shift: float = 0.0):
        series = {k: fred(k, "2025-06-01") for k in PAR}
        self.date = min(s.index[-1] for s in series.values())
        self.par = {k: float(series[k].loc[:self.date].iloc[-1]) / 100 for k in PAR}
        T = np.array(list(PAR.values()))
        p = np.array([self.par[k] for k in PAR]) + shift
        grid = np.arange(0.5, 30.001, 0.5)
        pg = np.interp(grid, T, p)
        D = []
        for t, c in zip(grid, pg):
            if t <= 1.0 + 1e-9:                        # bills: a single payment
                D.append(1 / (1 + c / 2) ** (2 * t))
            else:                                      # par bond, semiannual coupons
                D.append((1 - c / 2 * sum(D)) / (1 + c / 2))
        short = [np.log(1 + p[0] / 12) * 12, np.log(1 + p[1] * 0.25) / 0.25]
        self.t = np.r_[1 / 12, 0.25, grid]
        self.z = np.r_[short, -np.log(np.array(D)) / grid]
        # Straight linear interpolation made the 3 month forwards jump 30 bp at the bill to
        # coupon seam around 2 years; NSS through the bootstrapped zeros removes it.
        p0 = [self.z[-1], self.z[0] - self.z[-1], 0.0, 0.0, 1.5, 8.0]
        lo, hi = [-0.1, -0.2, -0.3, -0.3, 0.05, 0.5], [0.2, 0.2, 0.3, 0.3, 10.0, 30.0]
        self.b = least_squares(lambda b: nss(self.t, b) - self.z, p0, bounds=(lo, hi)).x
        self.fit_error_bp = float(np.abs(nss(self.t, self.b) - self.z).max() * 1e4)

    def zero(self, t):
        return nss(t, self.b)

    def df(self, t):
        t = np.asarray(t, float)
        return np.exp(-self.zero(t) * t)

    def fwd(self, t, h=1e-3):
        """Instantaneous forward rate."""
        t = np.asarray(t, float)
        a = np.maximum(t - h, 1e-6)
        return -(np.log(self.df(t + h)) - np.log(self.df(a))) / (t + h - a)
