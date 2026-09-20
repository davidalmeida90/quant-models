"""quantlib, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install kagglehub pandas pyarrow
import kagglehub
path = kagglehub.dataset_download("dudesurfin/spy-options-eod-volatility-surface-2010-2023")
# -> spy_eod_2010.parquet ... spy_eod_2023.parquet, 14 files, about 600 MB

# the layout is WIDE: one row per quote date, expiry and strike, carrying BOTH sides,
# with bracketed headers that you strip on load or every reference gets ugly
#   [QUOTE_DATE] [EXPIRE_DATE] [DTE] [UNDERLYING_LAST] [STRIKE]
#   [C_BID] [C_ASK] [C_IV] [C_DELTA] [C_GAMMA] [C_VEGA] [C_THETA]
#   [P_BID] [P_ASK] [P_IV] [P_DELTA] [P_GAMMA] [P_VEGA] [P_THETA]

import pandas as pd
df = pd.read_parquet(f"{path}/spy_eod_2018.parquet")
df.columns = [c.strip().strip("[]") for c in df.columns]
