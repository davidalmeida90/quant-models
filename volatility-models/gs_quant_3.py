"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# gs_quant/content/reports_and_screens/00_fx/vol_screen_app.py, theirs
def volatility_screen(crosses, start_date, end_date, tenor='3m', plot=True):
    fxspot_dataset, fxvol_dataset = Dataset('FXSPOT'), Dataset('FXIMPLIEDVOL')
    for cross in crosses:
        spot = fxspot_dataset.get_data(start_date, end_date, bbid=cross)[['spot']]
        spot_data[cross] = volatility(spot['spot'], tenor)          # realized
        vol = fxvol_dataset.get_data(..., tenor=tenor, deltaStrike='DN', location='NYC')
        impvol_data[cross] = vol.drop_duplicates(keep='last') * 100  # implied

    spdata, ivdata = format_df(spot_data), format_df(impvol_data)
    diff = ivdata.subtract(spdata).dropna()
    ...
        data[cross] = {'Spot': last_value(spot_fx[cross]),
                       '{} Implied'.format(tenor): last_value(ivdata[cross]),
                       '{} Realized'.format(tenor): last_value(spdata[cross]),
                       'Diff': last_value(diff[cross]),
                       'Historical Implied Low': min(ivdata[cross]),
                       'Historical Implied High': max(ivdata[cross]),
                       '%-ile': last_value(percentiles(ivdata[cross]))}

g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD',
       'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']
