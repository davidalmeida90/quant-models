"""alpaca paper, from the write-up at
https://davidariasfinance.com/scripts/free-trading-tools/
"""

# pip install alpaca-py
# paper keys from alpaca.markets, Paper Trading, set as environment variables:
#   ALPACA_KEY, ALPACA_SECRET
import os
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

KEY, SECRET = os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"]

trading = TradingClient(KEY, SECRET, paper=True)     # routes to paper-api.alpaca.markets
acct = trading.get_account()
print(acct.status, acct.buying_power, acct.portfolio_value)

order = trading.submit_order(MarketOrderRequest(
    symbol="SPY", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
print(order.id, order.status)                       # accepted, then filled a moment later

data = StockHistoricalDataClient(KEY, SECRET)
bars = data.get_stock_bars(StockBarsRequest(
    symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=30), feed=DataFeed.IEX))   # IEX is the free feed
print(bars.df.tail())
