# QuantConnect Lean, local backtest with the lean CLI

Engine: [github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean), Apache 2.0. CLI:
[github.com/QuantConnect/lean-cli](https://github.com/QuantConnect/lean-cli). Command names below are from
the lean-cli README on 27 September 2026. Docker Desktop has to be running before `lean backtest`. No
QuantConnect account is needed for a local backtest.

```bash
pip install --upgrade lean
mkdir lean-workspace && cd lean-workspace
lean init                                        # sample data + lean.json, no login needed
lean create-project --language python "SPY Hold" # starter main.py and research.ipynb
lean backtest "SPY Hold"                         # runs in Docker, results in SPY Hold/backtests/
```

Replace the generated `SPY Hold/main.py` with [lean_spy_hold.py](lean_spy_hold.py) before the backtest.
It buys SPY once with the full portfolio and holds, which is the baseline every strategy has to beat.
