# Trading Integration: 5 Free Tools to Test a Strategy, plus Glassbench

Runnable code behind the write-up on connecting a strategy to a paper account: 2 broker APIs, a multi agent LLM framework, an MCP server that hands TradingView data to Claude, an open source backtesting engine, and the routing skeleton that ties a signal to a broker.

![Desk: Python engine on the left, Interactive Brokers TWS paper account on the right](https://davidariasfinance.com/scripts/free-trading-tools/assets/ibdesk_frame.jpg)

## Tools

| Tool | What it is | File here | Upstream |
|---|---|---|---|
| Alpaca | Broker API with a free paper account and IEX data, Python SDK alpaca-py | [alpaca_paper.py](alpaca_paper.py) | [alpacahq/alpaca-py](https://github.com/alpacahq/alpaca-py) |
| Interactive Brokers | TWS socket API on port 7497 for paper, through ib_async | [ibkr_paper.py](ibkr_paper.py) | [ib-api-reloaded/ib_async](https://github.com/ib-api-reloaded/ib_async) |
| TradingAgents | 12 LLM agents in 5 stages that read, debate and rate a stock | [tradingagents_run.py](tradingagents_run.py) | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) |
| Glassbench | Web desk that runs TradingAgents and AI Hedge Fund side by side and sends the decision to a paper broker | in its own repo | [davidalmeida90/glassbench](https://github.com/davidalmeida90/glassbench) |
| TradingView MCP | 37 screener and technical analysis tools for Claude, Cursor and any MCP client | [tradingview_mcp.md](tradingview_mcp.md), [tradingview_mcp.json](tradingview_mcp.json) | [atilaahmettaner/tradingview-mcp](https://github.com/atilaahmettaner/tradingview-mcp) |
| QuantConnect Lean | Open source backtesting and live engine, run locally with the lean CLI and Docker | [lean_commands.md](lean_commands.md), [lean_spy_hold.py](lean_spy_hold.py) | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) |

[route.py](route.py) is the loop that joins them: a signal function, a risk check, an order to Alpaca or IBKR, the fill read back, and a JSONL log with slippage in basis points.

## Run

```bash
pip install -r requirements.txt
```

Credentials only through environment variables, never in a file: `ALPACA_KEY` and `ALPACA_SECRET` for Alpaca paper, `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` for TradingAgents. IBKR needs no key, only TWS or IB Gateway logged into the paper account.

```bash
py -3 alpaca_paper.py                       # account, 1 SPY market order, 30 days of IEX daily bars
py -3 ibkr_paper.py                         # TWS paper on 7497: MES and SPY, 1 MES sell, fills, positions, PnL
py -3 tradingagents_run.py --provider deepseek --ticker NVDA --date 2026-09-01
py -3 tradingagents_run.py --provider ollama    # local, after: ollama pull qwen3:8b
BROKER=alpaca py -3 route.py                # routes the placeholder signal, appends 1 line to fills.jsonl
```

TradingAgents is a git install, so it is not in requirements.txt:

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents && pip install .
```

TradingView MCP and Lean are command line tools rather than scripts; the exact lines are in [tradingview_mcp.md](tradingview_mcp.md) and [lean_commands.md](lean_commands.md). `lean_spy_hold.py` runs inside the Lean Docker container, not in a local Python.

## Paper trading loop, 7 steps

1. Signal. Your model returns a side (BUY, SELL, or nothing) and the price it saw when it decided. Keep that price, it is the only way to measure slippage later.
2. Risk check, before anything touches a broker. Notional cap, a position limit, and a kill switch on daily loss.
3. Order through the API. Market orders for the first weeks, so fills are certain and the comparison is clean.
4. Fill. Read it back from the broker, from the execution log on IBKR or the order object on Alpaca. Never assume the order filled at the signal price.
5. Log. One line per fill with timestamp, side, quantity, signal price, fill price and slippage in basis points.
6. Compare against the backtest, weekly. Same dates, same signals, backtest P&L against paper P&L. Gap is the cost of latency, spread and data.
7. Only then, live, at a size where a bug costs less than the lesson.

## Write-up

Full explanation, with a screenshot of each tool and the 2 TWS settings people miss: [https://davidariasfinance.com/scripts/free-trading-tools/](https://davidariasfinance.com/scripts/free-trading-tools/)

## License

MIT, see [LICENSE](../LICENSE). Paper accounts only; nothing here is investment advice.
