"""Risk based allocation: HRP, HCAA, risk parity, minimum variance

Extracted from the published write-up at
https://davidariasfinance.com/scripts/risk-based-allocation/
Run: py -3 model.py
"""

# --- chart output ------------------------------------------------------------
# Notebook code calls plt.show(); here every figure is written to charts/ instead.
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
from pathlib import Path as _Path
_CHARTS = _Path(__file__).resolve().parent / "charts"
_CHARTS.mkdir(parents=True, exist_ok=True)
_fig_n = [0]
def _save_figure(*_a, **_k):
    _fig_n[0] += 1
    _plt.gcf().savefig(_CHARTS / f"figure_{_fig_n[0]:02d}.png", dpi=140, bbox_inches="tight", facecolor="white")
    _plt.close("all")
_plt.show = _save_figure

try:  # plotly figures are written to charts/ as interactive HTML instead of printed
    import plotly.graph_objects as _go
    def _plotly_show(self, *_a, **_k):
        _fig_n[0] += 1
        _out = _CHARTS / f"figure_{_fig_n[0]:02d}.html"
        self.write_html(_out, include_plotlyjs="cdn")
        if _out.stat().st_size > 2_000_000:   # animation frames blow the file up; site has the live version
            _out.unlink()
    _go.Figure.show = _plotly_show
except ImportError:
    pass

# -----------------------------------------------------------------------------

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.cluster.hierarchy import ClusterWarning
from pathlib import Path
import warnings; warnings.filterwarnings('ignore', category=ClusterWarning)
plt.rcParams['figure.figsize'] = (9,5); plt.rcParams['axes.grid'] = False
NAMES = {"SPY": "US Large Cap", "IWM": "US Small Cap", "EZU": "Eurozone Equity", "EWU": "UK Equity", "SHY": "US 1-3y Tsy", "IEI": "US 3-7y Tsy", "IEF": "US 7-10y Tsy", "TLT": "US 20y+ Tsy", "BWX": "Intl Sovereign", "EEM": "EM Equity", "GLD": "Gold"}
TICKERS = ["SPY", "IWM", "EZU", "EWU", "SHY", "IEI", "IEF", "TLT", "BWX", "EEM", "GLD"]


px = pd.read_csv(_Path(__file__).resolve().parent / 'data' / 'prices_part2.csv', index_col=0, parse_dates=True)[TICKERS].rename(columns=NAMES)
rets = px.pct_change().dropna()
print(f'{px.shape[1]} assets, {px.shape[0]} trading days: {px.index[0].date()} -> {px.index[-1].date()}')
pd.DataFrame({'ETF': TICKERS, 'asset class': [NAMES[t] for t in TICKERS]})

corr_full = rets.corr()   # full-sample structure for the overview
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios':[1.05,1]})
im = ax[0].imshow(corr_full.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax[0].set_xticks(range(len(corr_full))); ax[0].set_xticklabels(corr_full.columns, rotation=90, fontsize=8)
ax[0].set_yticks(range(len(corr_full))); ax[0].set_yticklabels(corr_full.columns, fontsize=8)
ax[0].set_title('Correlation matrix (full sample)', weight='bold'); fig.colorbar(im, ax=ax[0], fraction=.046)
dist = np.sqrt(0.5*(1-corr_full))
link_avg = sch.linkage(squareform(dist.values, checks=False), method='average')
sch.dendrogram(link_avg, labels=corr_full.columns.tolist(), leaf_rotation=90, ax=ax[1],
               color_threshold=0.7*max(link_avg[:,2]))
ax[1].set_title('Dendrogram (average linkage)', weight='bold'); plt.tight_layout(); plt.show()

# ---------- HRP: de Prado 2016 appendix (Python-3 / pandas-3 port) ----------
def getIVP(cov):
    ivp = 1./np.diag(cov.values if hasattr(cov,'values') else cov); return ivp/ivp.sum()
def getClusterVar(cov, items):
    c = cov.loc[items, items]; w = getIVP(c).reshape(-1,1)
    return float((w.T @ c.values @ w)[0,0])
def getQuasiDiag(link):
    link = link.astype(int); s = pd.Series([link[-1,0], link[-1,1]]); n = link[-1,3]
    while s.max() >= n:
        s.index = range(0, s.shape[0]*2, 2)
        df0 = s[s >= n]; i = df0.index; j = df0.values - n
        s[i] = link[j,0]; df0 = pd.Series(link[j,1], index=i+1)
        s = pd.concat([s, df0]).sort_index()
        s.index = range(s.shape[0])
    return s.tolist()
def getRecBipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx); clusters = [sortIx]
    while clusters:
        clusters = [c[j:k] for c in clusters for j,k in ((0,len(c)//2),(len(c)//2,len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i+1]
            v0, v1 = getClusterVar(cov,c0), getClusterVar(cov,c1)
            a = 1 - v0/(v0+v1)
            w[c0] *= a; w[c1] *= 1-a
    return w
def correlDist(corr): return ((1-corr)/2.)**0.5
def getHRP(cov, corr):
    dist = correlDist(corr)
    link = sch.linkage(squareform(dist.values, checks=False), 'single')
    sortIx = [corr.index[i] for i in getQuasiDiag(link)]
    return getRecBipart(cov, sortIx).reindex(cov.index)

# ---------- HCAA: Raffinot 2018 (equal-weight 50/50 down the dendrogram) ----------
def getHCAA(corr, method='average'):
    dist = np.sqrt(2*(1-corr))
    link = sch.linkage(squareform(dist.values, checks=False), method=method)
    tree = sch.to_tree(link); w = pd.Series(0.0, index=corr.index)
    def rec(node, wt):
        if node.is_leaf(): w.iloc[node.id] = wt
        else: rec(node.left, wt/2); rec(node.right, wt/2)
    rec(tree, 1.0); return w

# ---------- Min-Variance: classic Markowitz risk minimization (long-only, no returns) ----------
def getMinVar(cov):
    n = len(cov); S = cov.values*252; x0 = np.ones(n)/n
    r = minimize(lambda w: w @ S @ w, x0, method='SLSQP', bounds=[(0,1)]*n,
                 constraints=[{'type':'eq','fun':lambda w: w.sum()-1}])
    return pd.Series(r.x, index=cov.index)

def getEW(cols): return pd.Series(1/len(cols), index=cols)

# ---------- Naive Risk Parity = de Prado's IVP: w proportional to 1/variance, ignores correlation ----------
def getNaiveRP(cov): return pd.Series(getIVP(cov), index=cov.index)

LOOKBACK = 756   # trading days for vol/correlation: 252=1y, 756=3y, 1260=5y
WIN = rets.tail(LOOKBACK)
cov, corr = WIN.cov(), WIN.corr()
print(f'Estimation window: {LOOKBACK} days (~{LOOKBACK//252}y)  {WIN.index[0].date()} -> {WIN.index[-1].date()}')
W = pd.DataFrame({'Equal-Weight': getEW(rets.columns), 'Naive RP': getNaiveRP(cov),
                  'Min-Variance': getMinVar(cov), 'HRP': getHRP(cov, corr),
                  'HCAA': getHCAA(corr)}).reindex(rets.columns)
(W*100).round(1)

import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = 'notebook'
WINDOWS = [('6m',126),('1y',252),('2y',504),('3y',756),('5y',1260)]
MC = {'Equal-Weight':'#6b7480','Naive RP':'#7a5aa6','Min-Variance':'#b4443a','HRP':'#2b4c8c','HCAA':'#B0862B'}
assets = list(rets.columns)
def w_for(win):
    r = rets.tail(win); c, co = r.cov(), r.corr()
    return (pd.DataFrame({'Equal-Weight':getEW(r.columns),'Naive RP':getNaiveRP(c),
                          'Min-Variance':getMinVar(c),'HRP':getHRP(c,co),
                          'HCAA':getHCAA(co)}).reindex(rets.columns)*100)
def traces(win):
    W = w_for(win)
    return [go.Bar(name=m, y=assets, x=W[m].values, orientation='h', marker_color=MC[m]) for m in MC]
DEF = 3   # default frame = 3y
fig = go.Figure(data=traces(WINDOWS[DEF][1]))
fig.frames = [go.Frame(data=traces(w), name=lbl) for lbl,w in WINDOWS]
steps = [dict(method='animate', label=lbl, args=[[lbl], dict(mode='immediate',
             frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for lbl,_ in WINDOWS]
fig.update_layout(barmode='group', height=560, template='plotly_white', bargap=0.25,
    title='Final allocation vs estimation window, drag the slider', xaxis_title='weight %',
    legend=dict(orientation='h', y=1.07),
    sliders=[dict(active=DEF, currentvalue=dict(prefix='Window: ', font=dict(size=15)),
                  pad=dict(t=45), steps=steps)])
fig.show()

from scipy.stats import skew, kurtosis
def _metrics(win):
    dates = rets.resample('ME').last().index
    fns = {'Equal-Weight':lambda w:getEW(w.columns),'Naive RP':lambda w:getNaiveRP(w.cov()),
           'Min-Variance':lambda w:getMinVar(w.cov()),'HRP':lambda w:getHRP(w.cov(),w.corr()),
           'HCAA':lambda w:getHCAA(w.corr())}
    rows = {}
    for m, fn in fns.items():
        port={}; sspw=[]; cur=None
        for t in rets.index:
            if cur is not None: port[t]=float((cur*rets.loc[t]).sum())
            if t in dates:
                wd = rets.loc[:t].tail(win)
                if len(wd)>=win:
                    cur = fn(wd).reindex(rets.columns).fillna(0); sspw.append(float((cur**2).sum()))
        pr = pd.Series(port).sort_index(); eq=(1+pr).cumprod()
        ann=eq.iloc[-1]**(252/len(pr))-1; vol=pr.std()*np.sqrt(252); shp=(pr.mean()*252)/vol
        dd=(eq/eq.cummax()-1).min()
        mo=((1+pr).resample('ME').prod()-1).dropna(); S=skew(mo); K=kurtosis(mo,fisher=True)
        asr=shp*(1+(S/6)*shp-(K/24)*shp**2)
        rows[m]={'Return %':ann*100,'Vol %':vol*100,'MaxDD %':dd*100,'SSPW':float(np.mean(sspw)),
                 'Sharpe':shp,'Adj Sharpe':asr}
    return pd.DataFrame(rows).T.round(2)
WINDOWS = [('6m',126),('1y',252),('2y',504),('3y',756),('5y',1260)]
METR = {lbl: _metrics(win) for lbl,win in WINDOWS}
COLS = ['Method','Return %','Vol %','MaxDD %','SSPW','Sharpe','Adj Sharpe']
def table_for(lbl):
    d = METR[lbl]
    return go.Table(columnwidth=[1.4,1,1,1,1,1,1.2],
        header=dict(values=COLS, fill_color='#191936', font=dict(color='white', size=12), align='left'),
        cells=dict(values=[d.index.tolist()]+[d[c].tolist() for c in d.columns],
                   fill_color=[['#eef1fa']*5]+[['#ffffff']*5]*6, align='left', height=30))
figT = go.Figure(data=[table_for('3y')])
figT.frames=[go.Frame(data=[table_for(lbl)], name=lbl) for lbl,_ in WINDOWS]
stepsT=[dict(method='animate',label=lbl,args=[[lbl],dict(mode='immediate',frame=dict(duration=0,redraw=True))]) for lbl,_ in WINDOWS]
figT.update_layout(height=400, margin=dict(t=55,b=10),
    title='Backtest performance vs estimation window, drag the slider',
    sliders=[dict(active=3, currentvalue=dict(prefix='Window: '), pad=dict(t=20), steps=stepsT)])
figT.show()

fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
colors = {'Equal-Weight':'#6b7480','Naive RP':'#7a5aa6','Min-Variance':'#b4443a','HRP':'#2b4c8c','HCAA':'#B0862B'}
for i, m in enumerate(W.columns):
    s = W[m].sort_values(ascending=True)
    ax[i].barh(range(len(s)), s.values*100, color=colors[m])
    ax[i].set_yticks(range(len(s))); ax[i].set_yticklabels(s.index, fontsize=8)
    effN = 1/(W[m]**2).sum()
    ax[i].set_title(f'{m}   (effective N = {effN:.1f})', weight='bold')
    ax[i].set_xlabel('weight %')
for j in range(len(W.columns), len(ax)): ax[j].axis('off')   # hide the empty 6th panel
plt.tight_layout(); plt.show()
conc = pd.DataFrame({'max weight %': (W.max()*100).round(1),
                     'sum sq weights (SSPW)': (W**2).sum().round(3),
                     'effective N': (1/(W**2).sum()).round(1)})
conc

COST_BPS = 10   # one-way transaction cost per unit of turnover (bps)
def backtest(rets, alloc_fn, lookback=LOOKBACK, rebal='ME'):
    dates = rets.resample(rebal).last().index
    w_prev = pd.Series(0.0, index=rets.columns); port = []; turn = {}; sspw = []
    cur_w = None
    for t in rets.index:
        if cur_w is not None:
            port.append((t, float((cur_w * rets.loc[t]).sum())))
        if t in dates:
            win = rets.loc[:t].tail(lookback)
            if len(win) >= lookback:
                cur_w = alloc_fn(win).reindex(rets.columns).fillna(0)
                turn[t] = float((cur_w - w_prev).abs().sum())
                sspw.append(float((cur_w**2).sum()))
                w_prev = cur_w
    return pd.Series(dict(port)).sort_index(), pd.Series(turn), float(np.mean(sspw))

def apply_cost(pr, turn, bps):
    net = pr.copy(); c = bps/1e4
    for d, tv in turn.items():
        if d in net.index: net.loc[d] -= c*tv
    return net
def perf(pr):
    eq = (1+pr).cumprod(); ann = eq.iloc[-1]**(252/len(pr))-1
    vol = pr.std()*np.sqrt(252)
    return eq, ann*100, vol*100, (pr.mean()*252)/vol, (eq/eq.cummax()-1).min()*100

allocs = {
  'Equal-Weight': lambda w: getEW(w.columns),
  'Naive RP': lambda w: getNaiveRP(w.cov()),
  'Min-Variance': lambda w: getMinVar(w.cov()),
  'HRP': lambda w: getHRP(w.cov(), w.corr()),
  'HCAA': lambda w: getHCAA(w.corr()),
}
curves, stats, prets = {}, {}, {}
for m, fn in allocs.items():
    pr, turn, sspw = backtest(rets, fn)
    eq, cg, vg, sg, ddg = perf(pr)               # gross
    _, c10, _, s10, _ = perf(apply_cost(pr, turn, 10))
    _, c25, _, s25, _ = perf(apply_cost(pr, turn, 25))
    curves[m] = (1+apply_cost(pr, turn, COST_BPS)).cumprod(); prets[m] = pr   # equity curve NET of COST_BPS
    stats[m] = {'CAGR %': cg, 'Vol %': vg, 'Sharpe': sg, 'MaxDD %': ddg, 'SSPW': sspw,
                'Turn/reb': turn.mean(), 'Ann turn': turn.mean()*12,
                'Net CAGR 10bp': c10, 'Net Shrp 10bp': s10,
                'Net CAGR 25bp': c25, 'Net Shrp 25bp': s25}
C = pd.DataFrame(curves); S = pd.DataFrame(stats).T
S.T.round(2)   # metrics as rows, the five methods as columns, so every column fits on screen

fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                       gridspec_kw={'height_ratios':[2,1]})
for m in C.columns:
    ax[0].plot(C.index, C[m], label=m, color=colors[m], lw=1.8)
ax[0].set_yscale('log'); ax[0].set_ylabel('growth of $1 (log)'); ax[0].legend(); 
ax[0].set_title(f'Out-of-sample equity curves, NET of {COST_BPS} bps costs ({LOOKBACK}d lookback, monthly rebalance)', weight='bold')
for m in C.columns:
    dd = C[m]/C[m].cummax()-1
    ax[1].plot(dd.index, dd*100, color=colors[m], lw=1.2)
ax[1].set_ylabel('drawdown %'); ax[1].set_title('Drawdown', weight='bold')
plt.tight_layout(); plt.show()

from scipy.stats import skew, kurtosis
def adj_sharpe(pr, sr):
    m = ((1+pr).resample('ME').prod()-1).dropna()   # monthly returns for skew/kurtosis
    S = skew(m); K = kurtosis(m, fisher=True)        # K = excess kurtosis
    return sr*(1 + (S/6)*sr - (K/24)*sr**2)
P = pd.DataFrame({m: {'Return % (CAGR)': stats[m]['CAGR %'], 'Volatility %': stats[m]['Vol %'],
                      'Max Drawdown %': stats[m]['MaxDD %'], 'Concentration (SSPW)': stats[m]['SSPW'],
                      'Sharpe': stats[m]['Sharpe'],
                      'Adjusted Sharpe': adj_sharpe(prets[m], stats[m]['Sharpe'])}
                  for m in allocs}).T
P.T.round(3)   # metrics as rows, methods as columns, fits fully on screen

import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = 'notebook'
def _bt_eq(window, rebal='ME', bps=10):
    rdates = rets.resample(rebal).last().index; c = bps/1e4; out = {}
    for m, fn in allocs.items():
        w_prev = pd.Series(0.0, index=rets.columns); port={}; turn={}; cur=None
        for t in rets.index:
            if cur is not None: port[t]=float((cur*rets.loc[t]).sum())
            if t in rdates:
                wd = rets.loc[:t].tail(window)
                if len(wd)>=window:
                    cur = fn(wd).reindex(rets.columns).fillna(0)
                    turn[t]=float((cur-w_prev).abs().sum()); w_prev=cur
        pr = pd.Series(port).sort_index()
        for d,tv in turn.items():
            if d in pr.index: pr.loc[d]-=c*tv
        out[m] = (1+pr).cumprod()
    return out
WOPTS = [('1y',252),('3y',756),('5y',1260)]
ROPTS = [('Monthly','ME'),('Quarterly','QE'),('Semiannual','2QE'),('Annual','YE')]
COMBOS = [(f'{wl} | {rl}', w, rc) for wl,w in WOPTS for rl,rc in ROPTS]   # window-major order
EQ = {name: _bt_eq(w, rebal=rc) for name,w,rc in COMBOS}
def _tr(name):
    cv = EQ[name]
    return [go.Scatter(x=cv[m].index, y=cv[m].values, name=m, mode='lines',
                       line=dict(color=colors[m], width=1.9)) for m in allocs]
DEF = '3y | Monthly'
fig = go.Figure(data=_tr(DEF))
fig.frames = [go.Frame(data=_tr(name), name=name) for name,_,_ in COMBOS]
steps = [dict(method='animate', label=name, args=[[name], dict(mode='immediate',
          frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for name,_,_ in COMBOS]
fig.update_layout(height=600, template='plotly_white', yaxis_type='log',
    title='Backtest equity curves: estimation window x rebalance frequency (net of 10 bps)',
    yaxis_title='growth of $1 (log)', legend=dict(orientation='h', y=1.09),
    sliders=[dict(active=[n for n,_,_ in COMBOS].index(DEF),
                  currentvalue=dict(prefix='Window | Rebalance:  ', font=dict(size=15)),
                  pad=dict(t=55), steps=steps)])
fig.show()

import matplotlib.cm as cm
snap_years = [2010, 2013, 2016, 2019, 2022, 2025]
snap_dates = [rets.index[rets.index <= f'{y}-12-31'][-1] for y in snap_years
              if (rets.index <= f'{y}-12-31').sum() >= LOOKBACK]
methods = ['Equal-Weight','Naive RP','Min-Variance','HRP','HCAA']; assets = list(rets.columns)
acol = cm.tab20(np.linspace(0, 1, len(assets)))
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5)); axes = axes.ravel()
for ax, d in zip(axes, snap_dates):
    win = rets.loc[:d].tail(LOOKBACK); c, co = win.cov(), win.corr()
    W = pd.DataFrame({'Equal-Weight':getEW(win.columns),'Naive RP':getNaiveRP(c),
                      'Min-Variance':getMinVar(c),'HRP':getHRP(c,co),
                      'HCAA':getHCAA(co)}).reindex(assets).fillna(0)
    bottom = np.zeros(len(methods))
    for ai, a in enumerate(assets):
        vals = np.array([W.loc[a, m]*100 for m in methods])
        ax.bar(methods, vals, bottom=bottom, color=acol[ai], label=a, width=0.72)
        bottom += vals
    ax.set_title(str(d.date()), weight='bold'); ax.set_ylim(0, 100); ax.set_ylabel('weight %')
    ax.tick_params(axis='x', rotation=18, labelsize=8)
for ax in axes[len(snap_dates):]: ax.axis('off')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Portfolio composition every ~3 years', weight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.97]); plt.show()

from arch import arch_model
import warnings; warnings.filterwarnings('ignore')
_gcache = {}
def garch_cov(win, corr):
    key = (win.index[-1], len(win))
    if key in _gcache: return _gcache[key]
    vols = {}
    for a in win.columns:
        r = win[a].dropna().values * 100.0   # scale up for GARCH numerical stability
        try:
            res = arch_model(r, mean='Zero', vol='Garch', p=1, q=1, dist='normal', rescale=False).fit(disp='off')
            fc = res.forecast(horizon=1, reindex=False)
            vols[a] = float(np.sqrt(fc.variance.values[-1,0])) / 100.0
        except Exception:
            vols[a] = float(win[a].std())   # fall back to sample vol if a fit fails
    d = pd.Series(vols).reindex(win.columns); D = np.diag(d.values)
    Sig = pd.DataFrame(D @ corr.values @ D, index=win.columns, columns=win.columns)
    _gcache[key] = Sig; return Sig

g_allocs = {
  'Equal-Weight': lambda w: getEW(w.columns),
  'Naive RP': lambda w: getNaiveRP(garch_cov(w, w.corr())),
  'Min-Variance': lambda w: getMinVar(garch_cov(w, w.corr())),
  'HRP': lambda w: getHRP(garch_cov(w, w.corr()), w.corr()),
  'HCAA': lambda w: getHCAA(w.corr()),
}
g_curves = {}; g_stats = {}
for m, fn in g_allocs.items():
    pr, turn, sspw = backtest(rets, fn, lookback=756, rebal='ME')
    net = apply_cost(pr, turn, 10); eq = (1+net).cumprod()
    ann = eq.iloc[-1]**(252/len(net))-1; vol = net.std()*np.sqrt(252); shp = (net.mean()*252)/vol
    g_curves[m] = eq
    g_stats[m] = {'Return % (CAGR)': ann*100, 'Volatility %': vol*100, 'Sharpe': shp,
                  'Max Drawdown %': (eq/eq.cummax()-1).min()*100,
                  'Adjusted Sharpe': adj_sharpe(net, shp), 'Concentration (SSPW)': sspw}
G = pd.DataFrame(g_stats)   # metrics as rows, the five methods as columns
fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={'height_ratios':[2,1]})
for m in g_curves:
    ax[0].plot(g_curves[m].index, g_curves[m], label=m, color=colors[m], lw=1.8)
    dd = g_curves[m]/g_curves[m].cummax()-1
    ax[1].plot(dd.index, dd*100, color=colors[m], lw=1.2)
ax[0].set_yscale('log'); ax[0].legend(); ax[0].set_ylabel('growth of $1 (log)')
ax[0].set_title('GARCH(1,1) forecast-vol allocation, out-of-sample (3y window, monthly, net 10 bps)', weight='bold')
ax[1].set_ylabel('drawdown %'); ax[1].set_title('Drawdown', weight='bold')
plt.tight_layout(); plt.show()
G.round(3)
