"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

err = lambda: 100 * np.mean([abs(h.calibrationError()) for h in helpers])
before = err()
model.calibrate(...)
print(f"{before:.2f}% -> {err():.2f}%")     # 35.15% -> 10.29%. if it does not move, it did not run
