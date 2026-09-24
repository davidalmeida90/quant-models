# Delta hedging error, a Monte Carlo on gamma

Sell an at the money call, hedge it once a day with Black-Scholes delta, and measure what one day of
hedging earns or loses across 150,000 simulated price paths. Prices drift at a 2% risk-free
rate with realised volatility equal to implied, and the hedge's cash pays interest, so the hedge is right
on average and every cent of error comes from gamma meeting a discrete rebalance.

Full write up, step by step with the charts: https://davidariasfinance.com/scripts/delta-hedging-error-monte-carlo/

## Run

```
pip install -r requirements.txt
python delta_hedging_mc.py
```

Writes `figures/*.png` and `results.json`. Runs in about ten seconds on a laptop.

## Results

Short at the money call, sigma 18%, r 2%, hedged once a day. Spread is the standard
deviation of one day's hedge P&L for paths starting the day at the strike.

| Days to expiry | Option value | Gamma | Daily hedge error | Error / value |
|---|---|---|---|---|
| 10 | $1.47 | 0.1112 | $0.103 | 7.0% |
| 5 | $1.03 | 0.1573 | $0.147 | 14.2% |
| 2 | $0.65 | 0.2487 | $0.232 | 35.8% |
| 1 | $0.46 | 0.3518 | $0.346 | 75.8% |

Last trading day, spot opening at the strike, delta rebalanced N times in the session:

| Rebalances | Every | Spread of P&L |
|---|---|---|
| 1 | 390 min | $0.341 |
| 2 | 195 min | $0.251 |
| 4 | 98 min | $0.184 |
| 12 | 32 min | $0.110 |
| 26 | 15 min | $0.076 |
| 78 | 5 min | $0.044 |

## What the script does

1. Parameters, including the flat risk-free rate
2. Black-Scholes price, delta and gamma
3. Simulated paths with common random numbers, so maturities differ by time alone, and one day of
   hedging P&L including interest on the cash account
4. Spread of the error by moneyness, vectorised with `np.bincount`
5. Rebalancing frequency on the final day, against the 1 / sqrt(N) rule

## Limitations

Geometric Brownian motion, so no jumps and no fat tails. Realised volatility equals implied, which removes
the variance risk premium on purpose. No transaction costs, constant volatility, no skew, a flat rate.
One option and one strike.

## License

MIT
