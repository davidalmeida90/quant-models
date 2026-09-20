"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install gs-quant
# installs and imports with no credentials. 46 of its 48 timeseries functions
# run fully offline, see the table above.
import gs_quant.timeseries as ts
from gs_quant.timeseries import Window

ts.volatility(px, Window(63, 0))          # annualised, already in percent
ts.exponential_volatility(px, 0.94)       # EWMA, RiskMetrics decay
ts.correlation(a, b, Window(63, 0))       # rolling pairwise
ts.beta(a, b, Window(63, 0))              # a regressed on b
ts.percentiles(series, w=Window(504, 0))  # rank against its own trailing 2 years
ts.max_drawdown(px, Window(252, 0))
ts.smooth_spikes(px, 0.10)                # their own bad print filter

# no GsSession anywhere. nothing leaves the machine.
