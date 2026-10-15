# Bond Option Pricing

Pricing the option inside three real bonds with free data, as of 14 October 2026: a convertible, a callable and a putable. Companion code to the YouTube video *Convertible and Callable Bonds: Pricing the Hidden Options*.

| Bond | CUSIP | Type | Who owns the option | Models |
|---|---|---|---|---|
| DoorDash 0% due 15 May 2030 | 25809KAB1 | Convertible | holder: a call on the stock | bond + Black-Scholes, Tsiveriotis-Fernandes stock tree |
| Viking Cruises 9.125% due 15 Jul 2031 | 92676XAG2 | Callable | issuer: a call on the bond | yield to worst, interest rate tree with OAS |
| Strategy 0% due 1 Dec 2029 | 594972AS0 | Putable (also convertible) | holder: a put at 100 on 1 Jun 2028 | stock tree with a put rule |

Convertible = straight bond + call. Callable = straight bond − call. Putable = straight bond + put.

## Run

```bash
pip install -r ../requirements.txt
py -3 model.py
```

Prints every result below; the same output is saved in `output.txt`.

## Results

```
Treasury zero rates: 1y 4.48%  1.6y 4.66%  3.6y 4.89%  4.75y 4.94%

VIKING 9.125% 2031, last trade 104.50
  yield to Jul 2027 @102.281  5.899%
  yield to Jul 2028 @100      6.356%
  yield to maturity 2031      7.963%
  OAS 97 bp | straight 112.83 | callable 104.50 | call worth 8.33 points

STRATEGY 0% 2029, last trade 90.43, put 1 Jun 2028 @100, stock 143.24, vol 71%
  yield to put 6.37% | yield to maturity 3.26% | Treasury 1.6y 4.67% (cont.)
  implied credit spread 204 bp | straight bond 80.53
  without the put 83.11 -> put worth 7.32 | without conversion 89.35 -> conversion worth 1.08
  conversion value today 21.3

DOORDASH 0% 2030, last trade 94.48, stock 184.01, vol 48%, Treasury 4.89%
  conversion value 63.0 | premium 50%
  implied credit spread 181 bp -> bond floor 78.65
  Black Scholes: one call 48.09 $ -> x3.425 = 16.47 points -> bond + call = 95.12
  Tree (TF, soft call May 2028 at 130%): 94.48 | delta 0.1829 per 100 face = 1.83 shares per bond (53% of the 3.425 conversion shares)
```

## What each model does

- **Treasury curve.** Discount factors bootstrapped from the 14 Oct 2026 par yields, semiannual coupons on a half year grid, bills priced money market style.
- **Yield to worst (Viking).** Street convention, semiannual, 30/360, to each call date at its call price and to maturity. FINRA's last trade yield, 5.899%, is the yield to the July 2027 call at 102.281, the lowest of the three.
- **Bond + Black-Scholes (DoorDash).** Bond floor: 100 discounted at the Treasury rate plus the bond's own credit spread. Call: 3.425 Black-Scholes calls per 100 of face, strike $291.97, to May 2030.
- **Tsiveriotis-Fernandes tree (DoorDash).** Binomial stock tree, 1,000 steps. At each node the bond is worth the higher of holding and converting. Value expected in shares is discounted at the risk free rate, value expected in cash at the risk free rate plus the credit spread. DoorDash can call from 20 May 2028 when the stock is at or above 130% of the conversion price. Credit spread solved so the tree matches the FINRA price.
- **Interest rate tree (Viking).** Lognormal short rate tree in the style of Black-Derman-Toy, 20% rate volatility, fitted so it reprices the Treasury curve exactly. On each call date the bond is worth the lower of its value and the call price. Option adjusted spread solved to match 104.50; the same tree without the call rule gives the straight bond.
- **Putable (Strategy).** Same stock tree as DoorDash with one more rule: on 1 Jun 2028 the bond is worth the higher of holding it and 100. Credit spread solved to match 90.43, then the tree is rerun without the put and without the conversion to split the price.

## What is inside

- Files: `model.py`, `output.txt`
- Data: `data/bonds.csv` (terms, with the filing each one comes from, and the last FINRA trade), `data/treasury_par_yields_2026-10-14.csv`, `data/market_inputs.csv` (stock closes and option implied volatilities)

Every input is free:

- [FINRA Fixed Income Data](https://www.finra.org/finra-data/fixed-income), search by CUSIP, for trade prices.
- [SEC EDGAR](https://www.sec.gov/edgar/search/) for the terms: DoorDash 10-K note 9, Strategy 10-K note 8, Viking's indenture Section 3.07.
- [U.S. Treasury daily par yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) for rates.
- [Yahoo Finance](https://finance.yahoo.com) for stock prices and listed option volatilities.

## References

- [Black & Scholes (1973), The Pricing of Options and Corporate Liabilities](https://doi.org/10.1086/260062)
- [Tsiveriotis & Fernandes (1998), Valuing Convertible Bonds with Credit Risk](https://doi.org/10.3905/jfi.1998.408240)
- [Black, Derman & Toy (1990), A One-Factor Model of Interest Rates and Its Application to Treasury Bond Options](https://doi.org/10.2469/faj.v46.n1.33)

## License

MIT, see [LICENSE](../LICENSE). Educational content, not investment advice.
