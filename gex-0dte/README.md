# Same day SPX gamma, closing in on expiration

Dealer gamma exposure (GEX) on the full SPX chain, with the same day expiry aged hour by hour while spot,
vols and open interest stay fixed. It follows the gamma flip into the close, splits the same day book by
strike, sizes the hedge a small move really forces, lines up fifteen years of SqueezeMetrics GEX against
the next day's move, and measures how much the dealer sign convention changes the answer.

Full write up, step by step with the charts: https://davidariasfinance.com/scripts/gamma-exposure-gex/
The GEX trading bot that uses dealer gamma on the Interactive Brokers API: https://github.com/davidalmeida90/gex-trading-bot

## Run

```bash
pip install -r ../requirements.txt
py -3 gex_0dte.py
```

Run it during US market hours on a trading day, so the chain has a live same day expiry. Charts are
written to `charts/`, the numbers to `results.json`. Both sources are read at run time:

| Source | What it gives | Access |
|---|---|---|
| Cboe delayed quotes, `cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` | every SPX and SPXW contract: strike, expiry, open interest, implied vol | free, delayed |
| SqueezeMetrics, `squeezemetrics.com/monitor/static/DIX.csv` | daily SPX close and their GEX estimate since 2011 | free |

No market data is stored in this folder.

## What it does

| Step | Function | Output |
|---|---|---|
| 1. Chain with the same day expiry marked | `fetch_chain`, `same_day_vols` | 20,034 contracts on the 1 Oct 2026 run, 363 of them expiring that day |
| 2. Dealer gamma with the same day book aged | `dealer_gex`, `profile`, `flip` | `charts/gamma_flip_aging.png` |
| 3. Same day book by strike | `same_day_by_strike` | `charts/same_day_by_strike.png` |
| 4. Hedge for a 0.25% move against the 1% figure | `hedge_for_move`, `linear_one_pct` | `charts/hedge_vs_linear.png` |
| 5. SqueezeMetrics GEX against the next day | `squeeze_history`, `spreads` | `charts/gex_next_day.png` |
| 6. Sign convention scenarios | `sign_scenarios` | `results.json` |

## Results, 1 Oct 2026 at 13:38 New York

SPX at 7,662, same day open interest 402,723 contracts.

| Hours to expiration | 6.5 | 3 | 1 | 0.25 | 0.1 |
|---|---|---|---|---|---|
| Gamma flip | 7,680 | 7,680 | 7,682 | 7,684 | 7,686 |
| Dealer gamma at spot, $bn per 1% | -23.2 | -24.2 | -24.2 | -24.0 | -23.7 |

At the biggest same day strike, 7,650 with 6,177 contracts:

| Measure | 6.5 hours out | 0.1 hours out |
|---|---|---|
| Usual GEX figure, gamma x 1% move | $3.4bn | $27.4bn |
| Exact hedge for a 0.25% move | $0.82bn | $2.36bn |
| Whole strike, contracts x 100 x strike | $4.73bn | $4.73bn |

The 1% figure is a straight line drawn from gamma, and near expiry gamma only lives within a few points
of the strike, so the line overshoots by an order of magnitude. The exact hedge is the change in delta for
the move, which can never pass the strike's whole delta.

SqueezeMetrics, 3,876 days to 29 Sep 2026: the next day moved ±2.25% (standard deviation) on the 347 days
with negative GEX and ±0.89% on positive GEX days. Without 2020 and 2022 the negative GEX days still
moved ±1.77%.

## Assumptions

- **Sign.** Dealers are taken as long calls and short puts, the standard convention. Nobody outside the
  dealers sees their book, so every GEX number is an estimate. On this run, dealer gamma at spot is
  -23.2bn under the convention, -14.3bn if the same day flow nets to zero, and +31.0bn if customers sold
  the same day puts.
- **Open interest is from the morning.** Same day options opened and closed during the session are missing.
- **Ceteris paribus.** Aging the same day expiry holds spot, implied vols and every other expiry fixed.
- **Black-Scholes gamma on calendar time.** Vols are annualised on 365 days, so hours go over 8,760.
- **SqueezeMetrics GEX** is their own positioning estimate on their own scale, so it is not comparable
  with the chain figures above. The gap between negative and positive GEX days is an association.

MIT licensed, like the rest of the repository.
