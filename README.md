# Quant Models in Python

[![Star this repo](https://img.shields.io/github/stars/davidalmeida90/quant-models?style=for-the-badge&label=STAR%20THIS%20REPO&logo=github&color=133F92)](https://github.com/davidalmeida90/quant-models/stargazers)
[![Follow on X](https://img.shields.io/badge/Follow%20on%20X-@Davidariasfin-111111?style=for-the-badge&logo=x)](https://x.com/Davidariasfin)
[![Newsletter](https://img.shields.io/badge/Newsletter-free%20every%20week-C9A227?style=for-the-badge&logo=substack&logoColor=white)](https://davidariasfinance.com/newsletter/)
[![Write-ups](https://img.shields.io/badge/Write--ups-davidariasfinance.com-1E56B8?style=for-the-badge)](https://davidariasfinance.com/scripts/)

Open-source implementations of the quantitative finance models published on
[davidariasfinance.com](https://davidariasfinance.com/scripts/): option pricing and Greeks, volatility
surfaces, hedging, portfolio construction and Monte Carlo. Models run on free market data, on simulated
paths, or straight from the formulas. Every result on the site comes from the code in this repository,
and every folder links back to the full write-up.

**A star costs nothing and puts these models in front of other people. Following on
[X](https://x.com/Davidariasfin) is where every new model gets posted first.**

Machine learning and deep learning for finance, the GEX trading bot and the rest of the larger
projects keep their own repositories, listed in
[more models, in their own repositories](#more-models-in-their-own-repositories) below.

| Model | What it does | Data |
|---|---|---|
| [Black-Scholes](black-scholes/) | European prices, no-arbitrage bounds, implied volatility by Brent | analytic |
| [Option Greeks](option-greeks/) | Delta, gamma, theta, vega, rho, and how each behaves | analytic |
| [Second and third order Greeks](advanced-greeks/) | Vanna, volga, charm, veta, speed, zomma, color, ultima, checked against finite differences | live chain, yfinance |
| [Gamma surface](gamma-surface/) | Gamma across spot, strike, vol and time, plus the gamma and theta trade-off | simulated grid |
| [Heston volatility surface](heston-vol-surface/) | Stochastic volatility by characteristic function, inverted into an implied vol surface | simulated |
| [Neural network vol surface](neural-network-vol-surface/) | Two from-scratch numpy MLPs learn the SPY implied vol surface | real SPY chain, included |
| [Deep hedging](deep-hedging/) | A network learns to hedge a call spread, judged on tail risk | simulated paths |
| [Volga convexity spread](volga-convexity-spread/) | Where a short strangle's convexity sits, and how wing width changes it | smile fitted from a real chain |
| [Hierarchical risk parity](hierarchical-risk-parity/) | Lopez de Prado's HRP: cluster the correlation matrix, allocate down the tree | real prices, included |
| [Risk based allocation](risk-based-allocation/) | HRP, HCAA, risk parity, minimum variance and equal weight compared | real prices, included |
| [Mean variance optimization](mvo-portfolio-optimization/) | Efficient frontier, maximum Sharpe, constraints and a backtest | yfinance + FRED |
| [Monte Carlo, GBM](monte-carlo-gbm/) | Ten million paths, antithetic variates, convergence to the closed form | simulated paths |
| [CRR binomial tree](crr-binomial-tree/) | European and American options, early exercise, convergence | analytic |
| [Six volatility models](volatility-models/) | arch, QuantLib, FinancePy, tf-quant-finance, gs-quant and an SVI fit | free market data |
| [Time series momentum](time-series-momentum/) | 990 settings swept on 45 ETFs, in sample and out of sample | ETF prices, yfinance |
| [Yield curves, US and Brazil](yield-curve-us-brazil/) | FRED and Tesouro Direto, with a Nelson Siegel Svensson fit | FRED + BCB + Tesouro |

## Quickstart

```bash
# fork it on GitHub first, then
git clone https://github.com/<your-user>/quant-models.git
cd quant-models
pip install -r requirements.txt

cd gamma-surface
py -3 model.py          # charts land in charts/
```

## How each folder is organised

```
<model>/
  model.py        the engine, runnable on its own
  notebook.ipynb  the same model to explore step by step (where one exists)
  charts/         the figures the script produces
  data/           small input files, when the model needs any
  README.md       what it does, how to run it, references
```

## More models, in their own repositories

| Repository | What it is |
|---|---|
| [gex-trading-bot](https://github.com/davidalmeida90/gex-trading-bot) | Gamma exposure from the free Cboe chain, plus a last-half-hour MES strategy on the Interactive Brokers API |
| [machine-learning-trading-engine](https://github.com/davidalmeida90/machine-learning-trading-engine) | Gu, Kelly and Xiu (2020) reproduced on free data, with a paper-trading engine |
| [machine-learning-for-finance](https://github.com/davidalmeida90/machine-learning-for-finance) | Trees, random forests and XGBoost on a survivorship-free S&P 500 panel, plus PCA and Lasso |
| [deep-learning-for-finance](https://github.com/davidalmeida90/deep-learning-for-finance) | Five architectures on real market data: FFN, LSTM, transformer, CNN and autoencoder |
| [delta-hedging-error-monte-carlo](https://github.com/davidalmeida90/delta-hedging-error-monte-carlo) | Hedging error of a short call across 150,000 simulated paths |
| [spy-iv-surface](https://github.com/davidalmeida90/spy-iv-surface) | Deep learning the SPY implied volatility surface on 2010-2023 chains |
| [finance-agent-kit](https://github.com/davidalmeida90/finance-agent-kit) | Equity valuation skills and data MCP servers for DeepSeek Harness and Claude Code |

## Videos

- [Gamma Exposure (GEX) Trading Bot in Python with the IBKR API](https://www.youtube.com/watch?v=mA8H1k5O9vc)
- [Machine Learning Trading Bot in Python: Neural Network + IBKR API](https://www.youtube.com/watch?v=j-uwUwr3aXw)

More on the channel: [@DavidAriasCFA222](https://www.youtube.com/@DavidAriasCFA222)

## Contributing

Issues and pull requests are welcome: a new model, a clearer README, a bug in the maths, a data source
that broke. See [CONTRIBUTING.md](CONTRIBUTING.md).

New models are posted on X: [@Davidariasfin](https://x.com/Davidariasfin)

## License

MIT for the code, see [LICENSE](LICENSE). Charts and write-up text are CC BY 4.0, so credit
davidariasfinance.com when reusing them.

## Disclaimer

Educational and research material. Nothing here is investment advice, and none of it is a trading
recommendation.