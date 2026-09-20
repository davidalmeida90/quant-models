"""TIME SERIES MOMENTUM.  Moskowitz, Ooi & Pedersen (2012), JFE 104(2).

Their result: an instrument's own past return predicts its own next return, for
one to twelve months, and that persistence partially reverses at longer
horizons. Tested here on 45 liquid ETFs spanning the four asset classes they
use, from 2005 to today, with real adjusted closes.

Grid is lookback crossed with holding period. Positions are scaled by inverse
volatility, which is their construction, and the holding period is run as
overlapping books opened one day apart, which is the standard way to hold a
signal for K days without timing the entry.

Costs are charged. Two basis points of the traded notional every time a position
moves, which is generous for liquid ETFs and brutal for the corner of the grid
that wants to rebalance daily on a five day signal. Watching that corner go red
is most of the point.

Panels 1, 2 and 4 run the diversified basket. Panel 3 puts one dot per ETF,
which is the same cross section the paper reports.

    py -3 bt_ts_momentum.py --preview   |   py -3 bt_ts_momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import bt_data                                        # noqa: E402
import bt_panels as P                                 # noqa: E402
from ml_kit import cached                             # noqa: E402

NAME = "bt_ts_momentum"

LOOKBACK = np.unique(np.round(np.logspace(np.log10(5), np.log10(300), 34)).astype(int))
# Holding runs to 250 days rather than 120. The first sweep put its best
# cells hard against the 120 day ceiling, and a sweep whose optimum sits on
# the boundary has not finished sweeping.
HOLDING = np.unique(np.round(np.logspace(np.log10(1), np.log10(250), 36)).astype(int))
COST_BP = 2.0 / 1e4
VOL_WIN = 60
TARGET_VOL = 0.40          # per instrument, before the equal weight across them
POS_CAP = 3.0


def _roll_mean0(a, w):
    """Rolling mean down axis 0 over w rows. Rows before w-1 come back as zero."""
    c = np.cumsum(a, axis=0)
    out = np.zeros_like(a)
    out[w - 1:] = c[w - 1:]
    out[w:] -= c[:-w]
    return out / w


def _experiment():
    px = bt_data.etfs()
    names = list(px.columns)
    P_ = px.to_numpy(float)
    T, N = P_.shape
    R = np.zeros_like(P_)
    R[1:] = P_[1:] / P_[:-1] - 1.0
    R[~np.isfinite(R)] = 0.0
    live = np.isfinite(P_) & (np.arange(T)[:, None] > 0)

    # ex-ante vol, so nothing from the future sizes today's position
    v = np.sqrt(_roll_mean0(R ** 2, VOL_WIN) * 252.0)
    scale = np.clip(np.where(v > 1e-6, TARGET_VOL / np.maximum(v, 1e-6), 0.0),
                    0.0, POS_CAP)

    split = T // 2
    nl, nh = len(LOOKBACK), len(HOLDING)
    isr = np.zeros((nl, nh))
    oos = np.zeros((nl, nh))
    keep = {}
    isr_i = np.zeros((nl, nh, N))
    oos_i = np.zeros((nl, nh, N))
    best_pnl = None

    for a, L in enumerate(LOOKBACK):
        sig = np.zeros_like(P_)
        sig[L:] = np.sign(P_[L:] / P_[:-L] - 1.0)
        sig[~np.isfinite(sig)] = 0.0
        sig[~live] = 0.0
        for b, H in enumerate(HOLDING):
            pos = _roll_mean0(sig, int(H)) * scale        # overlapping books
            pnl = np.zeros_like(P_)
            pnl[1:] = pos[:-1] * R[1:]
            turn = np.zeros_like(P_)
            turn[1:] = np.abs(pos[1:] - pos[:-1])
            pnl -= COST_BP * turn
            pnl[~live] = np.nan

            port = np.nanmean(np.where(live, pnl, np.nan), axis=1)
            port = np.nan_to_num(port)
            isr[a, b] = P.sharpe(port[:split])
            oos[a, b] = P.sharpe(port[split:])
            for i in range(N):
                col = pnl[:, i]
                isr_i[a, b, i] = P.sharpe(col[:split])
                oos_i[a, b, i] = P.sharpe(col[split:])
            if best_pnl is None or isr[a, b] > best_pnl[0]:
                best_pnl = (isr[a, b], a, b, port.copy(), pnl.copy())
            keep[(a, b)] = port.copy()

    _, bi, bj, port, pnl_i = best_pnl
    # the same lookback held all the way to the top of the axis, so the two
    # curves differ in one thing only
    bj2 = len(HOLDING) - 1
    port2 = keep[(bi, bj2)]
    # one dot per ETF: its own best in sample, and that same rule out of sample
    sx, sy = [], []
    for i in range(N):
        k = int(np.argmax(isr_i[:, :, i]))
        a, b = np.unravel_index(k, isr_i.shape[:2])
        sx.append(isr_i[a, b, i]); sy.append(oos_i[a, b, i])

    # How much the 45 books actually overlap, at the winning setting. This is
    # the whole reason a diversified trend book beats any single leg of it:
    # the legs are close to independent, so the portfolio keeps the return
    # and sheds most of the risk.
    # Sharpe of each asset class sleeve, year by year, at the winning cell.
    # Answers when the strategy worked and what carried it, which the
    # correlation matrix could not.
    yrs = px.index.year.to_numpy()
    years = np.unique(yrs)
    order, blocks = bt_data.group_order(names)
    ymat = np.full((len(blocks), len(years)), np.nan)
    for gi, (lab, s0, s1) in enumerate(blocks):
        cols = order[s0:s1]
        sleeve = np.nanmean(pnl_i[:, cols], axis=1)
        for yj, y in enumerate(years):
            seg = sleeve[yrs == y]
            seg = seg[np.isfinite(seg)]
            if seg.size > 80 and seg.std(ddof=1) > 0:
                ymat[gi, yj] = seg.mean() / seg.std(ddof=1) * np.sqrt(252)

    W = pnl_i[:, order]
    ok = np.isfinite(W)
    W = np.where(ok, W, np.nan)
    C = np.full((len(order), len(order)), np.nan)
    for i in range(len(order)):
        for j in range(len(order)):
            m = np.isfinite(W[:, i]) & np.isfinite(W[:, j])
            if m.sum() > 250:
                C[i, j] = np.corrcoef(W[m, i], W[m, j])[0, 1]
    off = C[~np.eye(len(order), dtype=bool)]
    return dict(isr=isr, oos=oos, cum=np.cumsum(port),
                cum2=np.cumsum(port2), best2=np.array([bi, bj2]),
                best=np.array([bi, bj]), split=np.array(split),
                sx=np.array(sx), sy=np.array(sy),
                ymat=ymat, years=years,
                dates=np.array([d.strftime('%b %Y') for d in px.index]),
                corr=C, blocks=np.array([b[0] for b in blocks]),
                bstart=np.array([b[1] for b in blocks]),
                bstop=np.array([b[2] for b in blocks]),
                avg_off=np.array(np.nanmean(off)),
                names=np.array(names), n_inst=np.array(N))


E = cached(NAME, _experiment, version="5")
ISR, OOS = E["isr"], E["oos"]
BI, BJ = (int(v) for v in E["best"])
L_BEST, H_BEST = int(LOOKBACK[BI]), int(HOLDING[BJ])
IS_BEST, OOS_BEST = float(ISR[BI, BJ]), float(OOS[BI, BJ])
N_TRIALS = ISR.size
N_INST = int(E["n_inst"])
POS_FRAC = float((OOS > 0).mean()) * 100.0
BI2, BJ2 = (int(v) for v in E["best2"])
H2 = int(HOLDING[BJ2])
IS2, OOS2 = float(ISR[BI2, BJ2]), float(OOS[BI2, BJ2])
AVG_OFF = float(E["avg_off"])
BLOCKS = [(str(a), int(b), int(c)) for a, b, c in
          zip(E["blocks"], E["bstart"], E["bstop"])]
# Correlation between what a cell did in sample and what the SAME cell did
# out of sample, across all 891 of them. On a real effect it is high. On the
# pairs study, where nothing survived, it is not.
RANK_CORR = float(np.corrcoef(ISR.ravel(), OOS.ravel())[0, 1])
# What the paper prescribes: about a year of lookback, held about a month.
LI = int(np.argmin(np.abs(LOOKBACK - 252)))
HI = int(np.argmin(np.abs(HOLDING - 21)))
PAPER_OOS = float(OOS[LI, HI])

CFG = dict(
    title="TIME SERIES MOMENTUM: PARAMETER SWEEP",
    title_size=24, title_y=0.942,
    subtitle_y=0.916, subtitle_size=14.5,
    model_line=("long what rose over the lookback, short what fell, "
                "sized by inverse volatility, held for the holding period"),
    model_y=0.897, model_size=12.5,
    surf_title="IN-SAMPLE SHARPE SURFACE",
    surf_title_y=0.872, surf_title_size=13.5,
    subtitle=(f"time series momentum on {N_INST} ETFs  ·  2005 to 2026  ·  "
              "2bp costs  ·  Moskowitz, Ooi & Pedersen (2012)"),
    stats=[],
    p1=LOOKBACK, p2=HOLDING,
    p1_name="lookback, days", p2_name="holding, days",
    p1_ticks=[5, 20, 60, 300], p2_ticks=[1, 10, 50, 250],
    isr=ISR, oos=OOS, best=(BI, BJ),
    mat=E["ymat"], mat_rows=[b[0] for b in BLOCKS],
    mat_xticks=[(i, str(y)) for i, y in enumerate(E["years"])
                if int(y) % 5 == 0],
    mat_title="SHARPE BY YEAR, BY SLEEVE",
    mat_sub="red = the sleeve made money that year",
    dates=E["dates"],
    card=[("Trials", f"{N_TRIALS}"),
          ("Lookback", f"{L_BEST}d"),
          (None, ""),
          (f"hold {H_BEST}d", f"{IS_BEST:+.2f} / {OOS_BEST:+.2f}"),
          (f"hold {H2}d", f"{IS2:+.2f} / {OOS2:+.2f}")],
    surf_caption="",
    scat_x=E["sx"], scat_y=E["sy"], scat_size=34,
    scat_title="IN-SAMPLE vs OUT-OF-SAMPLE",
    scat_xlabel="in-sample Sharpe of its winner",
    cum=E["cum"], split=int(E["split"]),
    best2=(BI2, BJ2),
    cum2=E["cum2"],
    cum_label=f"held {H_BEST}d",
    cum2_label=f"held {H2}d",
    pnl_title=f"SAME {L_BEST} DAY SIGNAL, TWO HOLDING PERIODS",
    pnl_rows=[f"{H_BEST:>4}d   in {IS_BEST:+.2f}   out {OOS_BEST:+.2f}",
              f"{H2:>4}d   in {IS2:+.2f}   out {OOS2:+.2f}",
              f"{N_INST} ETFs, equal weight"],
    pnl_xlabel="trading days, 2005 to 2026",
    footer="",
    seconds=14.0,
)

if __name__ == "__main__":
    render_frame, total = P.build(CFG)
    P.run(render_frame, total, NAME)
