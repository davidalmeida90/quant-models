"""Heston calibrated to seven SPY smiles, against the single volatility Black-Scholes allows.

Characteristic function on Albrecher's branch, Lewis single integral on fixed Gauss Legendre
nodes, least squares on vega weighted errors. Prints the five parameters, the Feller check,
and the fit error of one Black-Scholes vol on the same quotes.

    py -3 heston_calibration.py

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
from cboe_chain import np, pd, norm, chain, spot, r, forward, black, implied_vol

def heston_cf(u, T, v0, kappa, theta, sigma, rho):
    """Characteristic function of log(S_T / F), Albrecher's branch (the little trap).
    The textbook branch crosses the log's branch cut at long maturities and returns
    prices outside the no arbitrage bounds."""
    xi = kappa - rho * sigma * 1j * u
    d = np.sqrt(xi ** 2 + sigma ** 2 * (1j * u + u ** 2))
    g = (xi - d) / (xi + d)
    e = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((xi - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (xi - d) / sigma ** 2 * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v0)

# fixed Gauss Legendre nodes on [0, 400]: the optimiser sees a smooth function
x, w = np.polynomial.legendre.leggauss(400)
U, W = 200 * (x + 1), 200 * w

def heston_call(F, K, T, disc, *p):
    """Lewis single integral, vectorised over every strike of one expiry."""
    k = np.log(K / F)
    phi = heston_cf(U - 0.5j, T, *p)
    integrand = (np.exp(-1j * np.outer(k, U)) * phi / (U ** 2 + 0.25)).real
    return disc * (F - np.sqrt(F * K) / np.pi * (integrand @ W))


def smile_quotes(expiry):
    """Out of the money only, strikes every $5 between 0.82 and 1.10 of the forward,
    bid above 5 cents and spread under a quarter of the mid."""
    F, D, T = forward(expiry)
    g = chain[chain.expiry == expiry].copy()
    g = g[((g.kind == "C") & (g.strike >= F)) | ((g.kind == "P") & (g.strike < F))]
    g = g[(g.bid > 0.05) & ((g.ask - g.bid) / g.mid < 0.25) & (g.strike % 5 == 0)
          & (g.strike / F > 0.82) & (g.strike / F < 1.10)]
    g["F"], g["D"] = F, D
    g["iv"] = [implied_vol(m, F, k, T, D, c == "C") for m, k, c in zip(g.mid, g.strike, g.kind)]
    return g[(g.iv > 0.03) & (g.iv < 1.0)]

days = chain.drop_duplicates("expiry").set_index("expiry")["T"] * 365
days = days[days >= 7]
picked = []
for target in (21, 45, 75, 120, 180, 270, 365):       # seven expiries, a month to a year
    left = days.drop(picked)
    picked.append((left - target).abs().idxmin())
q = pd.concat([smile_quotes(e) for e in picked], ignore_index=True)
print(f"{len(q)} quotes on {q.expiry.nunique()} expiries")


from scipy.optimize import least_squares

d1 =(np.log(q.F / q.strike) + q.iv ** 2 * q["T"] / 2) / (q.iv * np.sqrt(q["T"]))
q["vega"] = q.D * q.F * norm.pdf(d1) * np.sqrt(q["T"])

def residuals(p):
    out = []
    for _, g in q.groupby("expiry"):
        F, D, T = g.F.iloc[0], g.D.iloc[0], g["T"].iloc[0]
        c = heston_call(F, g.strike.values, T, D, *p)
        model = np.where(g.kind == "C", c, c - D * (F - g.strike.values))
        out.append((model - g.mid.values) / g.vega.values)    # dollars to vol points
    return np.concatenate(out)

fit = least_squares(residuals, x0=[0.02, 2.0, 0.04, 0.8, -0.7], x_scale="jac",
                    bounds=([1e-4, 0.05, 1e-4, 0.05, -0.999], [0.5, 15, 0.5, 3, 0.5]))
v0, kappa, theta, sigma, rho = fit.x
print(f"vol today {np.sqrt(v0):.1%}, long run {np.sqrt(theta):.1%}, kappa {kappa:.2f}, "
      f"vol of vol {sigma:.2f}, rho {rho:.2f}, Feller {2 * kappa * theta > sigma ** 2}")

# Black-Scholes on the same ruler: the best single vol is the mean, its miss the std
print(f"one vol {q.iv.mean():.1%} misses by {100 * q.iv.std(ddof=0):.2f} vol points")
