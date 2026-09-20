"""Faithful reproduction of Lopez de Prado (2016) HRP — on HIS OWN simulated data.
Ports the paper's appendix code (A.3 numerical example + A.4 Monte Carlo) to
Python 3 / pandas 3. CLA is replaced by a long-only min-variance QP (same objective).
"""
import sys, random, time
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ClusterWarning
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=ClusterWarning)

HERE = Path(__file__).parent; FIG = HERE / "charts"; FIG.mkdir(exist_ok=True)

# ---- HRP, exactly as in the paper (Py3/pandas3 port) ----
def getIVP(cov, **kw):
    ivp = 1. / np.diag(cov.values if hasattr(cov, "values") else cov); return ivp / ivp.sum()
def getClusterVar(cov, cItems):
    cov_ = cov.loc[cItems, cItems]; w_ = getIVP(cov_).reshape(-1, 1)
    return float((w_.T @ cov_.values @ w_)[0, 0])
def getQuasiDiag(link):
    link = link.astype(int); sortIx = pd.Series([link[-1, 0], link[-1, 1]]); numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]; i = df0.index; j = df0.values - numItems
        sortIx[i] = link[j, 0]; df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]).sort_index(); sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()
def getRecBipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx); cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            c0, c1 = cItems[i], cItems[i+1]; a = getClusterVar(cov, c0); b = getClusterVar(cov, c1)
            alpha = 1 - a/(a+b); w[c0] *= alpha; w[c1] *= 1 - alpha
    return w
def correlDist(corr):
    return ((1 - corr) / 2.) ** .5

def getHRP(cov, corr):                       # paper's callback (numpy in, weights out, orig order)
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    link = sch.linkage(correlDist(corr).values, "single")
    sortIx = corr.index[getQuasiDiag(link)].tolist()
    return getRecBipart(cov, sortIx).sort_index()

def getMV(cov, **kw):                        # long-only min-variance (stands in for CLA min-var)
    V = np.asarray(cov, float); s = np.mean(np.diag(V)); V = V / s   # scale so SLSQP is well-conditioned
    n = V.shape[0]
    res = minimize(lambda w: w @ V @ w, np.repeat(1/n, n), method="SLSQP", bounds=[(0, 1)]*n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},), options={"ftol": 1e-12, "maxiter": 500})
    return res.x

# ---- A.3  generateData (numerical example) ----
def generateData_ex(nObs, size0, size1, sigma1):
    np.random.seed(12345); random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))
    cols = [random.randint(0, size0 - 1) for _ in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    return pd.DataFrame(x, columns=range(1, x.shape[1] + 1)), cols

# ---- A.4  generateData (Monte Carlo, with shocks) ----
def generateData_mc(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    cols = [random.randint(0, size0 - 1) for _ in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])
    return x, cols

# ============ NUMERICAL EXAMPLE (A.3) ============
nObs, size0, size1, sigma1 = 10000, 5, 5, .25
x, cols = generateData_ex(nObs, size0, size1, sigma1)
print("correlated pairs (source, copy):", [(j+1, size0+i) for i, j in enumerate(cols, 1)])
cov, corr = x.cov(), x.corr()
link = sch.linkage(correlDist(corr).values, "single")
sortIx = corr.index[getQuasiDiag(link)].tolist()
hrp = getRecBipart(cov, sortIx).reindex(cov.index)
ivp = pd.Series(getIVP(cov), index=cov.index)
mv  = pd.Series(getMV(cov.values), index=cov.index)

def heat(ax, m, labels, title):
    im = ax.imshow(m, cmap="RdYlBu_r", vmin=-1, vmax=1); ax.set_title(title, weight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); return im
fig, ax = plt.subplots(figsize=(5.5, 5)); im = heat(ax, corr.values, corr.columns, "Correlation (as-is)"); fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(FIG/"corr0.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.5)); sch.dendrogram(link, labels=corr.columns.tolist(), ax=ax); ax.set_title("Dendrogram", weight="bold"); fig.tight_layout(); fig.savefig(FIG/"dendro.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.5, 5)); im = heat(ax, corr.loc[sortIx, sortIx].values, sortIx, "Quasi-diagonalized"); fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(FIG/"corr1.png", dpi=110); plt.close(fig)
alloc = pd.DataFrame({"HRP": hrp, "IVP": ivp, "CLA/MinVar": mv}).loc[sortIx]
fig, ax = plt.subplots(figsize=(9, 4.5)); alloc.plot.bar(ax=ax, width=.8); ax.set_ylabel("weight"); ax.set_title("Allocations: HRP vs IVP vs CLA (min-var)", weight="bold"); fig.tight_layout(); fig.savefig(FIG/"alloc.png", dpi=110); plt.close(fig)

top5 = lambda s: s.sort_values(ascending=False)[:5].sum()*100
print(f"\nTop-5 concentration:  CLA/MinVar {top5(mv):.2f}%   HRP {top5(hrp):.2f}%   IVP {top5(ivp):.2f}%")
print("(paper reports CLA 92.66%, HRP 62.57%)")
print("HRP weights:\n", (hrp*100).round(2).to_string())

# ============ MONTE CARLO (A.4) ============
def hrpMC(numIters=1000, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22):
    methods = {"IVP": lambda c, r: getIVP(c), "HRP": lambda c, r: getHRP(c, r).values, "CLA/MinVar": lambda c, r: getMV(c)}
    stats = {k: [] for k in methods}; pointers = range(sLength, nObs, rebal)
    for _ in range(int(numIters)):
        xx, _ = generateData_mc(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        r = {k: [] for k in methods}
        for p in pointers:
            xin = xx[p-sLength:p]; cov_ = np.cov(xin, rowvar=0); corr_ = np.corrcoef(xin, rowvar=0)
            xout = xx[p:p+rebal]
            for k, f in methods.items():
                r[k].append(xout @ f(cov_, corr_))
        for k in methods:
            rr = np.concatenate(r[k]); stats[k].append(np.prod(1 + rr) - 1)
    return pd.DataFrame(stats)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
np.random.seed(0); random.seed(0)
t0 = time.time(); S = hrpMC(numIters=N); dt = time.time() - t0
var = S.var(); out = pd.DataFrame({"std": S.std(), "variance": var, "var vs HRP": var/var["HRP"] - 1})
print(f"\n=== Monte Carlo out-of-sample ({N} iters, {dt:.0f}s) ===")
print(out.round(4))
print("(paper: variance CLA 0.1157 > IVP 0.0928 > HRP 0.0671; HRP ~72% lower var than CLA)")
