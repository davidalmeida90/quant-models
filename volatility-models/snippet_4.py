"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

pool = []
for c in G10:
    f = pd.DataFrame({"p": percentiles(spd[c]),      # their percentiles, expanding history
                      "d": diff[c]}).dropna()        # forward minus trailing vol
    pool.append(f)
    print(c, f.p.corr(f.d).round(3))

al = pd.concat(pool)
for lo, hi in [(0,20), (20,40), (40,60), (60,80), (80,101)]:
    s = al[(al.p >= lo) & (al.p < hi)]
    print(f"{lo:3d}-{hi:3d}  n={len(s):6,}  {s.d.mean():+.2f} vol points")
