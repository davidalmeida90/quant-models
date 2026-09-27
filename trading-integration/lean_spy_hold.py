"""lean spy hold, from the write-up at
https://davidariasfinance.com/scripts/free-trading-tools/
"""

# pip install --upgrade lean          (see lean_commands.md; this file replaces the generated main.py)
# AlgorithmImports exists inside the Lean Docker container, not in a local Python
from AlgorithmImports import *

class SpyHold(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100_000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetBenchmark(self.spy)

    def OnData(self, data: Slice):
        if not self.Portfolio.Invested and data.ContainsKey(self.spy):
            self.SetHoldings(self.spy, 1.0)       # 100% of equity into SPY
            self.Debug(f"Bought SPY at {data[self.spy].Close}")
