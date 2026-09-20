"""financepy, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)**2 + sigma**2 ) )

#  a      vertical level, the floor of the curve
#  b      overall slope, how fast the wings rise
#  rho    tilt, negative leans the curve to the downside
#  m      horizontal shift of the minimum
#  sigma  how rounded the bottom is, 0 would give a sharp kink
