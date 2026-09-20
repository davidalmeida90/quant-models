"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install tf-quant-finance tensorflow tf-keras scipy numpy
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"   # else protobuf errors
import numpy as np, tensorflow as tf, tf_quant_finance as tff
from scipy.stats import norm

# no download here on purpose: the inputs below are the real 15 June 2023 SPY
# chain, hard coded so the benchmark reproduces without fetching anything.

# the real 15 June 2023 SPY chain, so this ties to the rest of the page
SPOT, VOL, T, R, Q = 442.61, 0.1191, 29 / 365, 0.05, 0.015


def tff_price(strikes):
    """Google's implementation. Every argument is an array of the same length."""
    n = len(strikes)
    return tff.black_scholes.option_price(
        volatilities=np.full(n, VOL),
        strikes=strikes,
        expiries=np.full(n, T),
        spots=np.full(n, SPOT),
        discount_rates=np.full(n, R),
        dividend_rates=np.full(n, Q),
        is_call_options=strikes >= SPOT).numpy()


def scipy_price(strikes):
    """The same closed form, written out, as the control."""
    F = SPOT * np.exp((R - Q) * T)
    df = np.exp(-R * T)
    s = VOL * np.sqrt(T)
    d1 = (np.log(F / strikes) + 0.5 * s * s) / s
    d2 = d1 - s
    call = df * (F * norm.cdf(d1) - strikes * norm.cdf(d2))
    put = df * (strikes * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(strikes >= SPOT, call, put)


for n in (1_000, 100_000, 1_000_000):
    k = np.linspace(300.0, 600.0, n)
    tff_price(k[:10])                                   # warm the graph, or you time the compile
    t0 = time.perf_counter(); a = tff_price(k)
    t1 = time.perf_counter(); b = scipy_price(k)
    t2 = time.perf_counter()
    print(f"{n:9d} options   tff {1000*(t1-t0):7.1f} ms   "
          f"scipy {1000*(t2-t1):7.1f} ms   max diff {np.abs(a-b).max():.1e}")
