"""route, from the write-up at
https://davidariasfinance.com/scripts/free-trading-tools/
"""

# pip install alpaca-py ib_async
# keys through the environment only: ALPACA_KEY and ALPACA_SECRET for Alpaca paper;
# IBKR needs TWS or IB Gateway logged into the paper account on port 7497
import json, os, time
from datetime import datetime, timezone

def route(signal_fn, symbol="SPY", qty=1, broker="alpaca", max_notional=50_000, log="fills.jsonl"):
    side, ref = signal_fn(symbol)                     # ("BUY" | "SELL" | None, price the model saw)
    if side is None or qty * ref > max_notional:      # risk check before any broker call
        return None
    if broker == "alpaca":
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tc = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
        o = tc.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide[side],
                                               time_in_force=TimeInForce.DAY))
        time.sleep(2); o = tc.get_order_by_id(o.id)
        fill = float(o.filled_avg_price or 0)
    else:                                             # "ibkr", TWS paper on 7497
        from ib_async import IB, Stock, MarketOrder
        ib = IB(); ib.connect("127.0.0.1", 7497, clientId=9)
        c = Stock(symbol, "SMART", "USD"); ib.qualifyContracts(c)
        ib.placeOrder(c, MarketOrder(side, qty)); ib.sleep(2)
        fill = ib.fills()[-1].execution.avgPrice; ib.disconnect()
    row = {"t": datetime.now(timezone.utc).isoformat(), "broker": broker, "symbol": symbol,
           "side": side, "qty": qty, "signal_px": ref, "fill_px": fill,
           "slip_bps": 1e4 * (fill - ref) / ref}      # signed, compare with the backtest weekly
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row

if __name__ == "__main__":
    def always_buy(symbol):                           # replace with your model
        return "BUY", 500.0
    print(route(always_buy, broker=os.environ.get("BROKER", "alpaca")))
