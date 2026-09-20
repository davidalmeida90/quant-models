"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install QuantLib
import QuantLib as ql

ql.HestonProcess(rf, div, spot, v0, kappa, theta, sigma, rho)
ql.HestonModel(process)
ql.AnalyticHestonEngine(model)          # semi closed form, fast enough to calibrate
ql.HestonModelHelper(tenor, cal, spot, strike, vol_quote, rf, div)
model.calibrate(helpers, ql.LevenbergMarquardt(), ql.EndCriteria(...), constraint)

# THE TRAP: model.params() returns theta, kappa, sigma, rho, v0
# NOT the constructor order v0, kappa, theta, sigma, rho. Unpacking positionally
# gives a plausible looking set of numbers that are completely wrong.
