import pytz
from datetime import datetime
import pandas as pd

from BusinessLogic.volatility_models import Chain, MidPrices

def chain_prepare(chain, rates, expiration_time='16:00:00', calc=False):
    new_york_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(new_york_tz).strftime("%H:%M:%S")
    current_date = pd.to_datetime(datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d"))

    chain['strikes'] = chain['strikes'] / 1000.0

    #temp fix
    chain['tau'] = ((pd.to_datetime(chain['expiration_dates']) -  pd.to_datetime('today')).dt.days)/365
    chain = chain[chain['tau']>0/365]

    chain_obj = Chain(chain['ticker'], chain['strikes'],
                   chain['option_types'], chain['bids'], chain['asks'], chain['bid_sizes'],
                   chain['ask_sizes'], chain['expiration_dates'], rates, MidPrices.mid_prices, expiration_time,
                   current_date, current_time, chain['volume'], calculate=calc)

    chain_obj._filter_itm_options()

    return chain_obj
