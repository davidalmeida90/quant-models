"""Runs the five models on data as of AS_OF and writes charts/facts.json plus 13 charts.

    pip install -r ../requirements.txt
    export SEC_USER_AGENT="Your Name your@email"      # the SEC API asks for one
    python run_all.py                                 # add --refresh to download again
"""
from __future__ import annotations

import json
import warnings

import callable_puttable
import convertible
import hull_white
import mbs
import merton
import style as s
from data import AS_OF

warnings.filterwarnings("ignore")


def main():
    print(f"data as of {AS_OF.date()}")
    print("1. callable and puttable bonds (BDT)")
    curve, cp = callable_puttable.run()
    print("2. convertible (Tsiveriotis-Fernandes)")
    cv = convertible.run(curve)
    print("3. MBS prepayment option")
    mb = mbs.run(curve)
    print("4. Merton")
    mt = merton.run(curve)
    print("5. Hull-White")
    hw = hull_white.run(curve)

    ba = mt["today"]["BA"]
    print(f"""
callable/puttable  straight {cp['straight']:.2f}, callable {cp['callable']:.2f}, puttable {cp['puttable']:.2f}
convertible        {cv['ticker']} value {cv['price']:.1f}, floor {cv['floor']:.1f}, parity {cv['parity']:.1f}
MBS                rates -1 pt: +{mb['down1_mbs']:.1f} (no refi +{mb['down1_no_refi']:.1f}); +1 pt: -{mb['up1_mbs']:.1f}
Merton             Boeing distance to default {ba['distance_to_default']:.1f}, low {mt['boeing_low_dd']:.2f} on {mt['boeing_low_date']}
Hull-White         a {hw['a']:.2f}, sigma {hw['sigma']:.2f}%, 5y zero off by {abs(hw['error_bp']):.1f} bp
results in {s.RESULTS}""")
    json.loads(s.FACTS.read_text())


if __name__ == "__main__":
    main()
