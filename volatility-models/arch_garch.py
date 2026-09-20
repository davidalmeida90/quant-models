"""arch garch, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install kagglehub pandas numpy yfinance
# the forward each strike implies, from put call parity: C - P = e^(-rT)(F - K)
w = d[(abs(np.log(d.STRIKE / spot))  0) & (d.P_BID > 0)].copy()
w["F_k"] = w.STRIKE + (w.cmid - w.pmid) / np.exp(-r * T)
F = float(w.F_k.median())
q = r - np.log(F / spot) / T          # the implied yield. on SPX this IS the dividend

# note the difference q_SPY - q_SPX is independent of r, since r cancels. so the
# comparison below does not depend on getting the rate right, only on using the same one.
