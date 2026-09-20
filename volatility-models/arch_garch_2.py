"""svi fit, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install arch yfinance pandas numpy
import numpy as np, pandas as pd, yfinance as yf
from arch import arch_model


def returns(ticker="SPY", start="2015-01-01", end="2026-08-01"):
    """Percent log returns. arch wants percent, not decimals, or the optimiser struggles."""
    px = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=True)["Close"].squeeze()
    return (100 * np.log(px / px.shift(1)).dropna())


def fit_garch(r, dist="t"):
    """GARCH(1,1) with Student t errors. The t matters: equity returns have fat tails and
    a normal likelihood pushes alpha too low to compensate."""
    return arch_model(r, vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")


def describe(res):
    p = res.params
    persist = p["alpha[1]"] + p["beta[1]"]
    return {
        "alpha": p["alpha[1]"],                       # reaction
        "beta": p["beta[1]"],                         # memory
        "persistence": persist,
        "half_life_days": np.log(0.5) / np.log(persist),
        "uncond_vol": np.sqrt(p["omega"] / (1 - persist) * 252),
    }


def forecast_vol(res, horizon=21):
    """Annualised vol path. horizon=21 is one option month."""
    f = res.forecast(horizon=horizon, reindex=False)
    return np.sqrt(f.variance.values[-1] * 252)


def realised_vol(r, window=21):
    """Trailing realised, for comparison. Same units as the forecast."""
    return (r / 100).rolling(window).std() * np.sqrt(252) * 100


if __name__ == "__main__":
    r = returns("SPY")
    res = fit_garch(r)
    d = describe(res)
    print(f"alpha {d['alpha']:.4f}   beta {d['beta']:.4f}")
    print(f"persistence {d['persistence']:.4f}   half life {d['half_life_days']:.1f} days")
    print(f"implied unconditional vol {d['uncond_vol']:.1f}%")
    print("21 day forecast:", forecast_vol(res, 21).round(1)[:5], "...")
