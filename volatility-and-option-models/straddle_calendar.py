"""Long and short straddle, and a calendar spread, priced off the live SPY chain.

Prints the straddle cost, the move it implies, its breakevens, the calendar cost, its best
case at the front expiry and the range where it makes money, then the daily time decay of
each calendar leg as the days pass.

    py -3 straddle_calendar.py

Write-up: https://davidariasfinance.com/scripts/volatility-and-option-models/
"""
from cboe_chain import np, pd, norm, chain, spot, r, forward, black, implied_vol

def mid(expiry, kind, K):
    return chain.query("expiry == @expiry and kind == @kind and strike == @K").mid.iloc[0]

front = chain.loc[(chain["T"] * 365 - 30).abs().idxmin(), "expiry"]   # 23 Oct, 31 days
back = chain.loc[(chain["T"] * 365 - 59).abs().idxmin(), "expiry"]    # 20 Nov, 59 days
K = 775.0                                              # strike listed on both expiries

straddle = mid(front, "C", K) + mid(front, "P", K)
print(f"straddle {straddle:.2f}, implied move {straddle / spot:.2%}")
print(f"breakevens {K - straddle:.2f} and {K + straddle:.2f}")

Ff, Df, Tf = forward(front); Fb, Db, Tb = forward(back)
iv_front = implied_vol(mid(front, "C", K), Ff, K, Tf, Df)
iv_back = implied_vol(mid(back, "C", K), Fb, K, Tb, Db)
calendar = mid(back, "C", K) - mid(front, "C", K)

# value of the calendar on the day the front month expires, across spot
spot_grid = np.linspace(0.92 * K, 1.08 * K, 400)
left = Tb - Tf
back_value = black(spot_grid * np.exp(r * left), K, left, np.exp(-r * left), iv_back)
pnl = back_value - np.maximum(spot_grid - K, 0) - calendar
print(f"calendar {calendar:.2f}, best case {pnl.max():.2f}")
print(f"profitable between {spot_grid[pnl > 0].min():.0f} and {spot_grid[pnl > 0].max():.0f}")


def theta_per_day(K, T, vol):
    d1 = (vol * vol * T / 2 + r * T) / (vol * np.sqrt(T)); d2 = d1 - vol * np.sqrt(T)
    return (-K * norm.pdf(d1) * vol / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365

for days_gone in (0, 15, 25, 30):
    f = -theta_per_day(K, Tf - days_gone / 365, iv_front)
    b = -theta_per_day(K, Tb - days_gone / 365, iv_back)
    print(f"day {days_gone:2d}: front loses {f:.2f}, back loses {b:.2f}, calendar keeps {f - b:.2f}")
