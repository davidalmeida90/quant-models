"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# their format_df, kept, because for their own inputs it is correct
def format_df(data_dict):
    df = pd.concat(data_dict, axis=1)
    df.columns = data_dict.keys()
    return df.ffill().dropna()

spdata = format_df(spot_data)          # their line, safe here
ivdata = spdata.shift(-63)             # forward leg built AFTER the align, never ffilled
asof = ivdata.dropna(how="any").index[-1]   # last date the forward window is complete
diff = (ivdata - spdata).dropna(how="any")
