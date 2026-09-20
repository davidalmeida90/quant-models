"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install financepy kagglehub pandas numpy pyarrow
from pathlib import Path
import numpy as np, pandas as pd
import kagglehub
from financepy.utils import Date
from financepy.market.curves import DiscountCurveFlat
from financepy.products.equity import EquityVanillaOption
from financepy.utils.global_types import OptionTypes
from financepy.models.black_scholes import BlackScholes

# ---------- curation ----------
d["cmid"] = (d.C_BID + d.C_ASK) / 2
d["pmid"] = (d.P_BID + d.P_ASK) / 2
d = d[d.mid > 0.05]                        # a 1 cent option carries no information
d = d[(d.spread / d.mid)  0) & (d.P_BID > 0)].copy()
df = np.exp(-0.05 * T)
w["F_k"] = w.STRIKE + (w.cmid - w.pmid) / df    # each strike implies a forward
F = float(w.F_k.median())                        # median, so one wide quote cannot move it
q = 0.05 - np.log(F / spot) / T                  # back out the dividend yield

# ---------- invert, then the greeks ----------
val, exp = Date(15, 6, 2023), Date(15, 6, 2023).add_days(29)
disc, divc = DiscountCurveFlat(val, 0.05), DiscountCurveFlat(val, q)

for _, x in d.iterrows():
    call = x.STRIKE >= spot
    o = EquityVanillaOption(exp, float(x.STRIKE),
                            OptionTypes.EUROPEAN_CALL if call else OptionTypes.EUROPEAN_PUT)
    iv = o.implied_volatility(val, spot, disc, divc, float(x.mid))   # the inverse problem
    m = BlackScholes(iv)
    o.value(val, spot, disc, divc, m)
    o.delta(val, spot, disc, divc, m)
    o.gamma(val, spot, disc, divc, m)
    o.vega(val, spot, disc, divc, m)
    o.theta(val, spot, disc, divc, m)
