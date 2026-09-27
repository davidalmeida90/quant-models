"""ibkr paper, from the write-up at
https://davidariasfinance.com/scripts/free-trading-tools/
"""

# pip install ib_async
# TWS or IB Gateway logged into the paper account, and under Global Configuration > API > Settings:
#   Enable ActiveX and Socket Clients ticked, Read-Only API unticked, 127.0.0.1 in Trusted IPs
from ib_async import IB, Future, Stock, MarketOrder

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=7)      # 7497 = TWS paper, 4002 = IB Gateway paper

mes = Future("MES", "202612", "CME")           # Micro E-mini S&P 500, December 2026, $5 a point
spy = Stock("SPY", "SMART", "USD")
ib.qualifyContracts(mes, spy)                  # fills conId, exchange, multiplier
print(mes.localSymbol, mes.multiplier)         # MESZ6 5

trade = ib.placeOrder(mes, MarketOrder("SELL", 1))
ib.sleep(2)
print(trade.orderStatus.status, trade.orderStatus.avgFillPrice)

for f in ib.fills():                           # read fills here, not from orderStatus
    print(f.execution.time, f.execution.side, f.execution.shares, f.execution.avgPrice)

for p in ib.positions():
    print(p.contract.localSymbol, p.position, p.avgCost)

account = ib.managedAccounts()[0]              # DU... on paper
ib.reqPnL(account)
ib.sleep(1)
print(ib.pnl())                                # dailyPnL, unrealizedPnL, realizedPnL

ib.disconnect()
