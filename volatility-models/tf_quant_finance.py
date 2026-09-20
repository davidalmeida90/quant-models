"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install tf-quant-finance tensorflow tf-keras
# TensorFlow 2.21 supports Python 3.10 to 3.13 ONLY. On 3.14 pip reports
# "no matching distribution found", which reads like a broken index.
import tf_quant_finance as tff

tff.black_scholes.option_price(...)          # vanilla European, vectorised
tff.black_scholes.implied_vol(...)           # inverts the above, also vectorised
tff.models.heston.HestonModel(...)           # stochastic vol, Monte Carlo or PDE
tff.math.pde.fd_solvers.solve_backward(...)  # the finite difference engine
tff.math.random.multivariate_normal(...)     # low discrepancy sequences for MC

# every argument is an ARRAY. one option and a million options are the same call.
