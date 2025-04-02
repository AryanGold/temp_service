import asyncio
import time
from scipy.interpolate import LSQUnivariateSpline
from datetime import datetime
from functools import lru_cache
from py_vollib_vectorized import vectorized_black as black
import numpy as np
from copy import deepcopy
import pandas as pd
from numpy import nan
from functools import partial
from numpy import ndarray, array, arange, zeros, ones, argmin, minimum, maximum, clip
from numpy.linalg import norm
from numpy.random import normal
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from warnings import filterwarnings

from .kalman_help import estimate_Q_from_eq37
from .helpers import make_json_serializable

def tau(end_date, end_time=None, start_date=None, start_time=None, days_in_year=365, date_format='%Y-%m-%d'):
    # Parse end dates and times
    time_format = '%H:%M:%S'

    end_date_obj = datetime.strptime(end_date, date_format)

    if end_time is None:
        end_time_obj = datetime.strptime('00:00:00', time_format).time()
    else:
        end_time_obj = datetime.strptime(end_time, time_format).time()

    end_datetime_obj = datetime.combine(end_date_obj, end_time_obj)

    # If no start date is provided, use the current date
    if start_date is None:
        start_date_obj = datetime.now()
    else:
        # Parse start date
        start_date_obj = datetime.strptime(start_date, date_format).date()

    # If no start time is provided, use '00:00:00'
    if start_time is None:
        start_time_obj = datetime.strptime('00:00:00', time_format).time()
    else:
        # Parse start time
        start_time_obj = datetime.strptime(start_time, time_format).time()

    # Combine start date and time into a datetime object
    start_datetime_obj = datetime.combine(start_date_obj, start_time_obj)

    # Calculate date and time difference
    diff = end_datetime_obj - start_datetime_obj

    # Convert to fractions of a year
    seconds_in_a_year = 86400 * days_in_year
    fraction_of_a_year = diff.total_seconds() / seconds_in_a_year

    return fraction_of_a_year


class L2:
    def __init__(self):
        self.place_holder = ''

    def vamp_bid(self):
        return self.place_holder

    def vamp_ask(self):
        return self.place_holder

    def best_bid(self):
        return self.place_holder

    def best_ask(self):
        return self.place_holder


class Chain:
    def __init__(self, tickers, strikes, option_types, bid_prices, ask_prices, bid_sizes, ask_sizes, expiration_dates,
                 rates, mid_price_method, expiration_times=None, snap_shot_dates=None, snap_shot_times='00:00:00',
                 volume=0,
                 calculate=True, othercols=None, reinitialize=False):



        chain = pd.DataFrame(
            {'ticker': tickers, 'snap_shot_dates': snap_shot_dates, 'snap_shot_times': snap_shot_times,
             'expiration_dates': expiration_dates, 'expiration_times': expiration_times, 'strikes': strikes,
             'option_types': option_types, 'bid_prices': bid_prices,
             'bid_sizes': bid_sizes, 'ask_sizes': ask_sizes,
             'ask_prices': ask_prices,
             'rates': rates,

             'volume': volume, 'filter': False})
        chain['symbol'] = chain['ticker'].astype(str) +chain['expiration_dates'].astype(str) + chain['strikes'].astype(str) + chain['option_types'].astype(str)

        # chain['tau'] = chain.apply(lambda row: tau(row['expiration_dates'],  # Todo: Consider putting this in calculate
        #                                           row.expiration_times,
        #                                           row['snap_shot_dates'],
        #                                           row['snap_shot_times']), axis=1)

        if reinitialize == False:
            chain['snap_shot_datetimes'] = pd.to_datetime(
                chain['snap_shot_dates'].astype(str) + ' ' + chain['snap_shot_times'].astype(str))
            chain['expiration_dates'] = pd.to_datetime(chain['expiration_dates'])
            chain['snap_shot_dates'] = pd.to_datetime(chain['snap_shot_dates'])
            chain['mid_prices'] = (chain['bid_prices'] + chain['ask_prices']) / 2
            chain['tau'] = (pd.to_datetime(chain['expiration_dates'].astype(str) + ' ' + chain['expiration_times']) -
                            chain['snap_shot_datetimes']).dt.total_seconds() / (365.25 * 24 * 3600)

        if othercols is not None:
            cols = [i for i in othercols.columns if i not in chain.columns]
            chain_to_concat = othercols[cols]

            chain = pd.concat([chain, chain_to_concat], axis=1)

        input_chain = chain.copy(deep=True)

        # It may be necessary to filter out NaN or other non-numeric values as well.
        # chain = chain[chain.ask_prices > 0.0]
        # chain = chain[chain.bid_prices > 0.0]

        self.input_chain = input_chain
        self.chain = chain

        self._mid_price_method = mid_price_method
        if calculate:
            self.__calculate__()

    def __calculate__(self):
        chain = self.chain

        self.chain = self._mid_price_method(self.chain)
        self._forward_price()
        self._filter_itm_options()
        self._implied_volatility()
        self._moneyness()
        self._log_moneyness()
        self._bsm_delta()

        return self

    def _forward_price(self):
        from numpy import exp
        chain = self.chain


        def _forward_price_inner(chain):
            # Create separate dataframes for calls and puts
            calls = chain[chain.option_types == 'c'].copy()
            puts = chain[chain.option_types == 'p'].copy()
        
            # Merge calls and puts on 'strikes', 'expiration_dates', 'tau', and 'rates'
            chain_merged = calls.merge(puts, on=['strikes', 'expiration_dates', 'tau', 'rates'],
                                       suffixes=('_call', '_put'))
        
            # Calculate mid absolute differences
            chain_merged['mid_abs_diffs'] = abs(chain_merged['mid_prices_call'] - chain_merged['mid_prices_put'])
        
            # Calculate the index of minimum differences by expiration date
            min_diff_idx = chain_merged.groupby('expiration_dates')['mid_abs_diffs'].idxmin()
        
        
            # Create a temporary dataframe with minimum differences
            min_diff_chain = chain_merged.loc[min_diff_idx]
        
            pd.set_option('display.max_columns', None)
        
            # Compute forward prices
            min_diff_chain['forward_prices'] = min_diff_chain.strikes + np.exp(
                min_diff_chain.rates * min_diff_chain.tau) * \
                                               (min_diff_chain['mid_prices_call'] - min_diff_chain['mid_prices_put'])
        
        
            # Merge the forward prices to the main chain DataFrame
            chain = chain.merge(
                min_diff_chain[['expiration_dates', 'forward_prices']], on='expiration_dates', how='left')
        
            return chain
        
        chain = chain.groupby(['ticker', 'snap_shot_dates', 'snap_shot_times']).apply(
            lambda x: _forward_price_inner(x)).reset_index(
            drop=True)
        
        self.chain = chain
        return self

        """
        def _forward_price_inner(chain):
            # Create separate dataframes for calls and puts
            calls = chain[chain.option_types == 'c'].copy()
            puts = chain[chain.option_types == 'p'].copy()

            # Merge calls and puts on 'strikes', 'expiration_dates', 'tau', and 'rates'
            chain_merged = calls.merge(puts, on=['strikes', 'expiration_dates', 'tau', 'rates'],
                                       suffixes=('_call', '_put'))

            # Calculate mid absolute differences
            chain_merged['mid_abs_diffs'] = abs(chain_merged['mid_prices_call'] - chain_merged['mid_prices_put'])

            # Find top three minimum absolute differences by expiration date
            top_min_diffs = chain_merged.groupby('expiration_dates')['mid_abs_diffs'].nsmallest(3).reset_index()

            # Filter data to only include these top three per expiration
            chain_top_min_diffs = chain_merged.loc[top_min_diffs['level_1']]

            # Calculate combined bid-ask spreads for top three minimum diffs
            chain_top_min_diffs['bid_ask_spread'] = (
                        chain_top_min_diffs['ask_prices_call'] - chain_top_min_diffs['bid_prices_call'] +
                        chain_top_min_diffs['ask_prices_put'] - chain_top_min_diffs['bid_prices_put'])

            # Select the index with the smallest bid-ask spread for each expiration
            idx_min_spread = chain_top_min_diffs.groupby('expiration_dates')['bid_ask_spread'].idxmin()

            # Create a dataframe with the best forward price candidate strikes
            min_diff_chain = chain_top_min_diffs.loc[idx_min_spread]

            # Compute forward prices
            min_diff_chain['forward_prices'] = min_diff_chain['strikes'] + np.exp(
                min_diff_chain['rates'] * min_diff_chain['tau']) * (min_diff_chain['mid_prices_call'] - min_diff_chain[
                'mid_prices_put'])

            # Merge the forward prices to the main chain DataFrame
            chain = chain.merge(min_diff_chain[['expiration_dates', 'forward_prices']], on='expiration_dates',
                                how='left')
            #print(chain['forward_prices'])
            return chain
        """

        # Usage within the class or script

        print("A5: ", chain.columns)

        chain = chain.groupby(['ticker', 'snap_shot_dates', 'snap_shot_times']).apply(
            lambda x: _forward_price_inner(x)).reset_index(drop=True)



        self.chain = chain
        return self

    def _filter_itm_options(self):
        from pandas import concat

        if 'mid_prices_call' not in self.chain.columns:
            self.chain = self._mid_price_method(self.chain)

        if 'forward_prices' not in self.chain.columns:
            self._forward_price()

        chain = self.chain

        calls = chain[chain.option_types == 'c'].copy()
        puts = chain[chain.option_types == 'p'].copy()

        calls = calls[calls.strikes >= calls.forward_prices]
        puts = puts[puts.strikes < puts.forward_prices]

        chain_otmf = pd.concat([calls, puts]).sort_values(['tau', 'strikes']).reset_index(drop=True)

        self.chain = chain_otmf

    def _implied_volatility(self):
        from numba import NumbaDeprecationWarning
        from warnings import filterwarnings
        from numpy import errstate

        if 'mid_prices_call' not in self.chain.columns:
            self.chain = self._mid_price_method(self.chain)

        if 'forward_prices' not in self.chain.columns:
            self._forward_price()

        filterwarnings('ignore', category=NumbaDeprecationWarning)  # There is a problem with the vec vollib library

        from py_vollib_vectorized import vectorized_implied_volatility_black as iv_black

        chain = self.chain

        # Todo: Turn this all into a singe iv_black function call

        try:
            with errstate(divide='raise', invalid='ignore'):

                chain['bid_iv'] = iv_black(chain.bid_prices,
                                           chain.forward_prices,
                                           chain.strikes,
                                           chain.rates,
                                           chain.tau,
                                           chain.option_types,
                                           return_as='array')
        except ZeroDivisionError:
            chain['bid_iv'] = nan

        try:
            with errstate(divide='raise', invalid='ignore'):

                chain['ask_iv'] = iv_black(chain.ask_prices,
                                           chain.forward_prices,
                                           chain.strikes,
                                           chain.rates,
                                           chain.tau,
                                           chain.option_types,
                                           return_as='array')
        except ZeroDivisionError:

            chain['ask_iv'] = nan

        try:
            with errstate(divide='raise', invalid='ignore'):

                # chain['mid_iv'] = iv_black(chain.mid_prices,
                #                           chain.forward_prices,
                #                           chain.strikes,
                #                           chain.rates,
                #                           chain.tau,
                #                           chain.option_types,
                #                            return_as='array')

                chain['mid_iv'] = iv_black(chain['mid_prices'], chain['forward_prices'], chain['strikes'],
                                           chain['rates'], chain['tau'], chain['option_types'], return_as='array')

        except ZeroDivisionError:
            chain['mid_iv'] = nan

        chain = chain.sort_values(['tau', 'strikes'])
        self.chain = chain
        return self

    def _moneyness(self):
        chain = self.chain

        chain['moneyness'] = chain.strikes / chain.forward_prices

        self.chain = chain

    def _log_moneyness(self):  # Todo: Consider calculating this based of moneyness()
        from numpy import log
        chain = self.chain

        chain['moneyness'] = chain['strikes'] / chain['forward_prices']
        chain['log_moneyness'] = log(chain.moneyness)

        self.chain = chain

    def _bsm_delta(self):
        from py_vollib_vectorized import vectorized_delta

        if 'mid_iv' not in self.chain.columns:
            self._implied_volatility()

        chain = self.chain

        chain['delta'] = vectorized_delta(chain.option_types, chain.forward_prices, chain.strikes, chain.tau,
                                          chain.rates, chain.mid_iv)
        self.chain = chain
        return self

    def delta_buckets(self):  # Todo: Make the delta buckets a param so they can be changed
        from pandas import cut

        chain = self.chain
        bins = [-float('inf'), -50, -0.3, -0.1, 0, 0.1, 0.3, 0.5, float('inf')]
        labels = ['-inf to -50', '-.30--.50', '-.30--.10', '-.10-0', '0-.10', '.10-.30', '.30-.50', '50+']
        chain['delta_bucket'] = cut(chain.delta, bins=bins, labels=labels, right=False)
        self.chain = chain

    def filter_DataFrame(self, column, operator, value):
        chain = self.chain

        operators = {
            '==': chain[column] == value,
            '>': chain[column] > value,
            '>=': chain[column] >= value,
            '<': chain[column] < value,
            '<=': chain[column] <= value,
            '!=': chain[column] != value,
        }

        try:
            chain = chain[operators[operator]].reset_index(drop=True)
            self.chain = chain

        except KeyError:
            valid_operators = ", ".join(operators.keys())
            raise ValueError(f"Invalid operator '{operator}'. Valid operators are: {valid_operators}")

    def arbitrage_free_mid_vols(self, by_expiry=True):
        from arbitrage_repair import constraints, repair
        #import time
        from py_vollib_vectorized import vectorized_implied_volatility, vectorized_black
        chain = self.chain.sort_values(['ticker', 'snap_shot_dates', 'expiration_dates', 'strikes']).reset_index(
            drop=True)
        #s = time.time()

        def get_arb_free_mid_for_tick(chain_tick):
            '''

            import time
            s = time.time()


            puts = chain_tick[chain_tick['option_types'] == 'p']
            calls = chain_tick[chain_tick['option_types'] == 'c']

            # Calculate equivalent call prices from put prices using put-call parity
            # C = P + S - K * exp(-rT)
            calculated_call_mids = puts['mid_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
               -puts['rates'] * puts['tau'])
            calculated_call_bids = puts['bid_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
                -puts['rates'] * puts['tau'])
            #calculated_call_asks = puts['ask_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
                -puts['rates'] * puts['tau'])

            # Create a new DataFrame for the calculated call prices from puts
            derived_calls = pd.DataFrame({
                'strikes': puts['strikes'],
                'mid_prices': calculated_call_mids,
                'bid_prices': calculated_call_bids,
                'ask_prices': calculated_call_asks,
                'forward_prices': puts['forward_prices'],
                'rates': puts['rates'],
                'tau': puts['tau'],
                'ticker': puts['ticker']
            }, columns=['strikes', 'mid_prices', 'bid_prices', 'ask_prices', 'forward_prices', 'rates', 'tau',
                        'ticker'])

            # Concatenate derived call prices with existing call options
            combined_calls = pd.concat([calls, derived_calls], ignore_index=True)

            # Now you have combined call options where 'combined_calls' includes both existing and derived call prices
            call_mids = combined_calls['mid_prices']
            call_bids = combined_calls['bid_prices']
            call_asks = combined_calls['ask_prices']
            '''

            call_mids = vectorized_black(flag='c',
                                         F=chain_tick['forward_prices'],
                                         t=chain_tick['tau'],
                                         K=chain_tick['strikes'],
                                         r=chain_tick['rates'],
                                         sigma=chain_tick['mid_iv'],return_as='array')

            call_bids = vectorized_black(flag='c',
                                         F=chain_tick['forward_prices'],
                                         t=chain_tick['tau'],
                                         K=chain_tick['strikes'],
                                         r=chain_tick['rates'],
                                         sigma=chain_tick['bid_iv'],return_as='array')

            call_asks = vectorized_black(flag='c', F=chain_tick['forward_prices'], t=chain_tick['tau'],
                                         K=chain_tick['strikes'], r=chain_tick['rates'], sigma=chain_tick['ask_iv'],return_as='array')

            normaliser = constraints.Normalise()
            normaliser.fit(chain_tick['tau'].to_numpy(), chain_tick['strikes'].to_numpy(), call_mids,
                           chain['forward_prices'].to_numpy())

            T1, K1, C1 = normaliser.transform(chain['tau'].to_numpy(), chain['strikes'].to_numpy(),
                                              call_mids)

            mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)

            bid_spread = call_mids - call_bids
            ask_spread = call_asks - call_mids
            spreads = np.array([ask_spread, bid_spread])

            epsilon = repair.l1ba(mat_A, vec_b, C1, spreads)

            K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon)

            call_mids_arb_free = C0

            chain_tick['mid_iv_arb_free'] = vectorized_implied_volatility(flag='c',
                                                                          S=chain_tick['forward_prices'],
                                                                          t=chain_tick['tau'],
                                                                          K=chain_tick['strikes'],
                                                                          r=0,
                                                                          price=call_mids_arb_free)

            return chain_tick

        if by_expiry:
            grouper = ['ticker', 'snap_shot_datetimes', 'expiration_dates']
        else:
            grouper = ['snap_shot_datetimes', 'ticker']

        chain = chain.groupby(grouper).apply(lambda x: get_arb_free_mid_for_tick(x)).reset_index(drop=True)
        #print(time.time() - s)
        self.chain = chain

    def calculate_needed_col(self, needed_cols, args_dict={}, kwargs_dict={}):
        '''
        example_usage:

       needed_columns = ['mid_iv', 'log_moneyness']

       args_dict = {
           'mid_iv': (['input for arg1'],),  # Correct arguments
       }

       kwargs_dict = {
           'mid_iv': {'example_kwarg': 'keyword_argument for iv'}
       }

       '''

        from inspect import signature

        required_methods = {
            'mid_iv': self._implied_volatility,
            'bid_iv': self._implied_volatility,
            'ask_iv': self._implied_volatility,
            'log_moneyness': self._log_moneyness,
            'delta_bucket': self.delta_buckets,
            'delta': self._bsm_delta,
            'moneyness': self._moneyness,
            'forward_prices': self._forward_price,
            'mid_iv_arb_free': self.arbitrage_free_mid_vols
        }

        for col in needed_cols:
            if col not in self.chain.columns:
                method = required_methods[col]
                method_args = args_dict.get(col, ())
                method_kwargs = kwargs_dict.get(col, {})
                sig = signature(method)
                parameters = sig.parameters

                # Check for missing required arguments
                missing_args = [p for p in parameters.values()
                                if p.default == p.empty and  # Check if parameter is required
                                p.name not in method_kwargs and  # Not provided as a keyword argument
                                len(method_args) < list(parameters.keys()).index(
                        p.name) + 1]  # Not provided as a positional argument

                if missing_args:
                    missing_params = ', '.join(p.name for p in missing_args)
                    raise ValueError(
                        f"Missing required arguments for method `{method.__name__}` when calculating '{col}': {missing_params}")

                method(*method_args, **method_kwargs)

    def calculate_atm_implied_volatility(self):
        chain = self.chain

        # Ensure forward prices and implied volatilities are calculated
        if 'forward_prices' not in chain.columns:
            self._forward_price()
        if 'mid_iv' not in chain.columns:
            self._implied_volatility()

        atm_iv_list = []

        # Group by expiration dates to find ATM strike for each expiration
        for expiration, group in chain.groupby('expiration_dates'):
            group['strike_diff'] = (group['strikes'] - group['forward_prices']).abs()
            atm_option = group.loc[group['strike_diff'].idxmin()]
            atm_iv = atm_option['mid_iv']
            atm_iv_list.append({'expiration_dates': expiration, 'atm_iv': atm_iv})

        # Create a DataFrame from the ATM implied volatilities
        atm_iv_df = pd.DataFrame(atm_iv_list)

        # Merge the ATM implied volatilities back to the main chain DataFrame
        chain = chain.merge(atm_iv_df, on='expiration_dates', how='left')

        self.chain = chain


class MidPrices(Chain):
    @staticmethod
    def mid_prices(chain):

        chain['mid_prices'] = (chain.bid_prices + chain.ask_prices) / 2.0

        return chain

    @staticmethod
    def weighted_mid_prices(chain):

        imbalances = chain.bid_sizes / (chain.bid_sizes + chain.ask_sizes)

        chain['mid_prices'] = imbalances * chain.ask_prices + (1 - imbalances) * chain.bid_prices

        return chain


class OptionFilters:
    def __init__(self, chain_object: Chain):
        self.chain_object = chain_object

    def max_width_filter_delta_buckets(self, scaler_of_average, delete=False):
        chain_object = self.chain_object

        chain_object.delta_buckets()
        chain = chain_object.chain

        def calc_break_col(sing_delta_bucket, scaler_of_average_inner):
            # sing_delta_bucket['breaking_max_ba_width'] = False
            sing_delta_bucket['imp_vol_ba_width'] = sing_delta_bucket['ask_iv'] - sing_delta_bucket['bid_iv']
            bound = sing_delta_bucket['imp_vol_ba_width'].mean() * scaler_of_average_inner
            sing_delta_bucket.loc[sing_delta_bucket['imp_vol_ba_width'] > bound, 'filter'] = True
            return sing_delta_bucket

        chain = chain.groupby(['tau', 'delta_bucket']).apply(
            lambda x: calc_break_col(x, scaler_of_average))\
            .reset_index(drop=True).sort_values(['tau', 'strikes']).reset_index(drop=True)

        if delete:
            chain = chain[chain['filter'] == False].reset_index(drop=True)

        self.chain_object.chain = chain

        return self

    def max_width_filter(self, scalar_of_average: int or float = 10.0):
        chain = self.chain_object.chain.copy()  # Ensure we are working on a copy
        chain['iv_spread'] = chain['ask_iv'] - chain['bid_iv']

        def process_group(group):
            avg_iv_spread = group['iv_spread'].mean()
            if isinstance(scalar_of_average, (int, float)):
                condition = group['iv_spread'] > scalar_of_average * avg_iv_spread
                group.loc[condition, 'filter'] = True
            return group

        chain = chain.groupby(['ticker', 'snap_shot_dates', 'expiration_dates']).apply(process_group).reset_index(
            drop=True)
        chain.drop('iv_spread', axis=1, inplace=True)

        self.chain_object.chain = chain


    def increasing_ask(self):
        chain = self.chain_object.chain

        def filter_(chain_expo):

            # Process calls
            calls = chain_expo[chain_expo['option_types'] == 'c'].copy()
            calls = calls.sort_values('strikes')

            for idx, row in calls.iterrows():
                if (calls.loc[calls['strikes'] < row['strikes'], 'ask_prices'] < row['ask_prices']).any():
                    calls.at[idx, 'filter'] = True

            # Process puts
            puts = chain_expo[chain_expo['option_types'] == 'p'].copy()
            puts = puts.sort_values('strikes')
            for idx, row in puts.iterrows():
                if (puts.loc[puts['strikes'] > row['strikes'], 'ask_prices'] < row['ask_prices']).any():
                    puts.at[idx, 'filter'] = True

            # Combine results back into the original DataFrame
            chain_expo.update(calls)
            chain_expo.update(puts)

            return chain_expo

        self.chain_object.chain = chain.groupby(['ticker','snap_shot_datetimes','expiration_dates']).apply(lambda x: filter_(x)).reset_index(drop=True)




    def filter_customer_orders(self, scaler_of_average, delete=False):
        self.chain_object.delta_buckets()
        chain = self.chain_object.chain

        def filter_customer_order_on_chain(sing_delta_bucket, scaler_of_average_inner):
            # sing_delta_bucket['customer_order'] = False
            sing_delta_bucket['bidasksizeavg'] = (sing_delta_bucket['bid_sizes'] + sing_delta_bucket['ask_sizes']) / 2
            bound = sing_delta_bucket['bidasksizeavg'].mean() * scaler_of_average_inner
            sing_delta_bucket.loc[sing_delta_bucket['bidasksizeavg'] < bound, 'filter'] = True
            return sing_delta_bucket

        chain = chain.groupby(['tau', 'delta_bucket']).apply(
            lambda x: filter_customer_order_on_chain(x, scaler_of_average)).reset_index(drop=True)

        if delete:
            chain = chain[chain['customer_order'] == False].reset_index(drop=True)

        self.chain_object.chain = chain
        return self

    def global_median_filter(self, delete=False):
        chain_object = self.chain_object
        chain = chain_object.chain

        global_median_iv = chain['mid_iv'].median()
        lower_bound = global_median_iv / 15
        upper_bound = global_median_iv * 15

        chain.loc[(chain['mid_iv'] < lower_bound) | (chain['mid_iv'] > upper_bound), 'filter'] = True

        if delete:
            chain = chain[chain['filter'] == False].reset_index(drop=True)

        self.chain_object.chain = chain
        return self

    def maturity_median_filter(self, delete=False):
        chain_object = self.chain_object
        chain = chain_object.chain

        def filter_by_maturity(group):
            median_iv = group['mid_iv'].median()
            lower_bound = 0.2 * median_iv
            upper_bound = 5 * median_iv
            group.loc[(group['mid_iv'] < lower_bound) | (group['mid_iv'] > upper_bound), 'filter'] = True
            return group

        chain = chain.groupby('expiration_dates').apply(filter_by_maturity).reset_index(drop=True)

        if delete:
            chain = chain[chain['filter'] == False].reset_index(drop=True)

        self.chain_object.chain = chain
        return self


    def sort_chain_in_order_of_liquidity(self):

        chain = self.chain_object.chain

        def get_representative_volatility(single_maturity_chain):
            # define variables
            strikes = single_maturity_chain['strikes']
            forward = single_maturity_chain['forward_prices'].iloc[0]

            # Find the index of the strike closest to the spot price
            idx_closest = (strikes - forward).abs().idxmin()

            # Get the implied volatility at the closest strike price
            imp_vol_closest = single_maturity_chain.loc[idx_closest, 'mid_iv']

            # Add a new column to chain with the implied volatility at the closest strike for this maturity
            single_maturity_chain['impl_vol_closest_atm_for_maturity'] = imp_vol_closest

            return single_maturity_chain

        chain = chain.reset_index(drop=True)

        chain = chain.groupby('tau').apply(get_representative_volatility).reset_index(drop=True)

        def calc_liquidity_ratio(single_maturity_chain):
            from numpy import exp, sqrt, mean

            # define forward price and other variables given by the inputted chain:
            forward_price = single_maturity_chain['forward_prices'].iloc[0]
            representative_volatility = single_maturity_chain['impl_vol_closest_atm_for_maturity'].iloc[0]
            tau = single_maturity_chain['tau'].iloc[0]
            strikes = single_maturity_chain['strikes']
            bids = single_maturity_chain['bid_prices']
            asks = single_maturity_chain['ask_prices']

            # calculate the strike coverage ratio for the maturity and add it to the chain:
            upper_bound_spot = forward_price * np.exp(2 * representative_volatility * np.sqrt(tau))
            lower_bound_spot = forward_price * np.exp(-2 * representative_volatility * np.sqrt(tau))

            strike_coverage_ratio = len(single_maturity_chain[(strikes <= upper_bound_spot) &
                                                              (strikes >= lower_bound_spot)]) \
                / len(single_maturity_chain)

            # single_maturity_chain['strike_coverage_ratio_for_maturity'] = strike_coverage_ratio

            avg_bid_ask_spread = mean((asks + bids) / 2)
            liquidity_ratio = strike_coverage_ratio / avg_bid_ask_spread
            single_maturity_chain['liquidity_ratio'] = liquidity_ratio

            return single_maturity_chain

        chain = chain.groupby('tau').apply(calc_liquidity_ratio).sort_values(
            ['liquidity_ratio'], ascending=False).reset_index(drop=True)

        self.chain_object.chain = chain

        return self

    # Todo: This needs to be changed so that it marks the chain.filter column to True where applicable
    def remove_strike_arbitrage_on_all_expos(self):
        chain = self.chain_object.chain

        from pandas import concat, DataFrame
        from numpy import polyfit, polyval

        def remove_strike_arbitrage(single_maturity_chain):

            def quadratic_fit(x, y):
                # Fit a quadratic polynomial to the data
                p = polyfit(x, y, 2)

                # Use the fitted polynomial to predict y values
                y_pred = polyval(p, x)

                # Return the predicted y values as a pandas series

                return y_pred

            def remove_strike_causing_binary_arb(puts_or_calls):
                puts_or_calls = puts_or_calls.sort_values('strikes').reset_index(drop=True)
                # (data / np.max(data)) * desired_max
                for idx, df_strike in puts_or_calls.iterrows():
                    neighboring_strikes_index = \
                        puts_or_calls.loc[puts_or_calls['strikes'] == df_strike['strikes']].index[0]

                    number_of_total_strikes_in_poly = 4

                    neighbor_range = range(max(0, neighboring_strikes_index - 1),
                                           min(len(puts_or_calls),
                                               neighboring_strikes_index + number_of_total_strikes_in_poly - 1))

                    neighbor_strikes_df = puts_or_calls.loc[neighbor_range]

                    # display(neighbor_strikes_df)

                    neighbor_strikes_df['iv_quadratic_mid'] = quadratic_fit(neighbor_strikes_df['strikes'],
                                                                            neighbor_strikes_df['mid_iv'])

                    neighbor_strikes_df['fit_residual'] = neighbor_strikes_df['mid_iv'] - \
                        neighbor_strikes_df['iv_quadratic_mid']

                    max_error = neighbor_strikes_df['fit_residual'].abs().max()
                    neighbor_strikes_df['scaled_error'] = (neighbor_strikes_df['fit_residual'].abs() / max_error) * .1

                    neighbor_strikes_df = neighbor_strikes_df.sort_values('strikes').reset_index(drop=True)

                    neighboring_strikes = list(neighbor_strikes_df['strikes'])

                    puts_or_calls.loc[puts_or_calls['strikes'].isin(neighboring_strikes), 'weight'] += \
                        list(neighbor_strikes_df['scaled_error'])

                try:
                    max_weight_idx = puts_or_calls['weight'].idxmax()
                    strikexrr = float(puts_or_calls.loc[max_weight_idx, 'strikes'])
                    puts_or_calls = puts_or_calls.drop(max_weight_idx)
                    print(f'removed strike {strikexrr}')
                    return puts_or_calls.sort_values('strikes').reset_index(drop=True)
                except KeyError:
                    return None

            def add_initial_weights_to_digital_arb_strikes(puts_or_calls_chain, digital_arb_strikes):
                for strike in digital_arb_strikes:
                    # add 0.5 weight to calls causing digital arbitrage
                    puts_or_calls_chain.loc[puts_or_calls_chain['strikes'] == strike, 'weight'] += 0.5
                return puts_or_calls_chain

            # separate puts and calls
            calls = single_maturity_chain[
                single_maturity_chain['option_types'] == 'c'].sort_values('strikes').reset_index(drop=True)
            puts = single_maturity_chain[
                single_maturity_chain['option_types'] == 'p'].sort_values('strikes').reset_index(drop=True)

            # calls arbs:
            # butterfly arbs:
            call_butterfly_condition = (calls['mid_prices'].shift() - calls['mid_prices']) - \
                                       ((calls['strikes'] - calls['strikes'].shift()) /
                                        (calls['strikes'].shift(-1) - calls['strikes'])) * \
                                       (calls['mid_prices'] - calls['mid_prices'].shift(-1))

            butterfly_arb_calls = [calls.loc[i - 1:i + 1] for i in calls[(call_butterfly_condition < 0)].index]

        # digital_arb_calls contains a list of dataframes with the strikes causing every digital arbitrage in the calls
            call_digital_condition = (calls['mid_prices'] - calls['mid_prices'].shift(-1)) / (
                        calls['strikes'].shift(-1) -
                        calls['strikes'])

            digital_arb_calls = [calls.loc[i:i + 1] for i in calls[(call_digital_condition >= 1) |
                                                                   (call_digital_condition <= 0)].index]

            while digital_arb_calls or butterfly_arb_calls:
                # set/reset weight check after removing strike
                calls['weight'] = 0

                # get a list of all call strikes causing arbitrage
                if digital_arb_calls and not butterfly_arb_calls:
                    arb_calls_strikes = list(pd.concat(digital_arb_calls)['strikes'])
                if not digital_arb_calls and butterfly_arb_calls:
                    arb_calls_strikes = list(pd.concat(butterfly_arb_calls)['strikes'])
                if digital_arb_calls and butterfly_arb_calls:
                    arb_calls_strikes = list(pd.concat([pd.concat(digital_arb_calls),
                                                     pd.concat(butterfly_arb_calls)])['strikes'])


            # add initial weights. 0.5 for each strike causing binary arbitrage, 0.25 for each strike causing butterfly
                calls = add_initial_weights_to_digital_arb_strikes(calls, arb_calls_strikes)
                # call function to add regression weights to each strike and remove the highest weighted strike
                calls = remove_strike_causing_binary_arb(calls)

                # condition to check if calls is empty:
                if calls is None:
                    calls = pd.DataFrame(columns=puts.columns)
                    break

                # recheck for arbitrage
                call_digital_condition = (calls['mid_prices'] - calls['mid_prices'].shift(-1)) / \
                                         (calls['strikes'].shift(-1) - calls['strikes'])

                digital_arb_calls = [calls.iloc[i:i + 1] for i in calls[(call_digital_condition >= 1)
                                                                        | (call_digital_condition <= 0)].index]

                call_butterfly_condition = (calls['mid_prices'].shift() - calls['mid_prices']) - \
                                           ((calls['strikes'] - calls['strikes'].shift()) /
                                            (calls['strikes'].shift(-1) - calls['strikes'])) * \
                                           (calls['mid_prices'] - calls['mid_prices'].shift(-1))

                butterfly_arb_calls = [calls.loc[i - 1:i + 1] for i in calls[(call_butterfly_condition < 0)].index]

                # loop will continue if either list is not empty

            # put arbs:
            put_digital_condition = (puts['mid_prices'].shift(-1) - puts['mid_prices']) / \
                                    (puts['strikes'].shift(-1) - puts['strikes'])
            digital_arb_puts = [puts.loc[i:i + 1] for i in puts[(put_digital_condition >= 1) |
                                                                (put_digital_condition <= 0)].index]

            while digital_arb_puts:
                puts['weight'] = 0

                digital_arb_puts_strikes = list(pd.concat(digital_arb_puts)['strikes'])

                puts = add_initial_weights_to_digital_arb_strikes(puts, digital_arb_puts_strikes) \
                    .sort_values('strikes').reset_index(drop=True)
                puts = remove_strike_causing_binary_arb(puts).sort_values('strikes').reset_index(drop=True)

                if puts is None:
                    puts = pd.DataFrame(columns=calls.columns)
                    break

                put_digital_condition = (puts['mid_prices'].shift(-1) - puts['mid_prices']) / \
                                        (puts['strikes'].shift(-1) - puts['strikes'])
                digital_arb_puts = [puts.loc[i:i + 1] for i in puts[(put_digital_condition >= 1) |
                                                                    (put_digital_condition <= 0)].index]

            if not any([digital_arb_calls, digital_arb_puts, butterfly_arb_calls]):
                single_maturity_chain = pd.concat([puts, calls])
                print('no arb')
                return single_maturity_chain

            # single_maturity_chain = remove_strike_arbitrage(single_maturity_chain)

            return pd.concat([puts, calls])

        chain = chain.groupby(['tau']).apply(lambda x: remove_strike_arbitrage(x)).reset_index(drop=True).sort_values(
            ['tau', 'strikes']).reset_index(drop=True)
        if 'weight' in chain.columns:
            chain = chain.drop('weight', axis=1)

        original_chain = self.chain_object.chain
        merged_chain = original_chain.merge(chain[['symbol']], on='symbol', how='left', indicator=True)

        # Set 'filter' = True for shared rows
        merged_chain['filter'] = merged_chain['_merge'] == 'both'

        # Drop the '_merge' column
        merged_chain = merged_chain.drop(columns=['_merge'])

        # Update the original chain with the new 'filter' values
        self.chain_object.chain = merged_chain


        #self.chain_object.chain = chain
        return self

    def min_width_filter(self):
        return self





class MultiPointModel:
    def __init__(self, chain_object: Chain, points, weightings=1.0):  # Todo: need to save spline points and such
        self.chain_object = chain_object

        chain = chain_object.chain

        self.splines = {}

        chain['weightings'] = weightings

        filtered_chain = chain[chain['filter'] == False]

        grouped_filtered_chain = filtered_chain.groupby('expiration_dates')

        self.chain = chain
        self.grouped_filtered_chain = grouped_filtered_chain
        self.points = points
        self.__fit()


    def __fit(self,alternative_iv=None):
        grouped_filtered_chain = self.grouped_filtered_chain
        chain = self.chain
        points = self.points

        points = str(points)

        stdev_points = {
            '13': np.array([-10, -5, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1.5, 2, 3]),
            '11': np.array([-4, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 4]),
            '9': np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
            '7': np.array([-3, -2, -1, 0, 1, 2, 3]),
            '5': np.array([-2, -1, 0, 1, 2])
        }

        for expiration, group in grouped_filtered_chain:

            log_moneyness = group.log_moneyness
            mid_iv = group.mid_iv
            weightings = group.weightings

            closest_to_atmf_idx = (log_moneyness - 0).abs().idxmin()
            atmf_iv = mid_iv[closest_to_atmf_idx]
            tau = group.tau[closest_to_atmf_idx]

            try:
                knot_points_stdev = stdev_points[points]

            except KeyError:
                raise ValueError(f'{points} is not a valid points parameter input, the valid parameters inputs are: '
                                 f'{stdev_points.keys()}')

            knot_points = pd.DataFrame({})
            knot_points['stdev'] = knot_points_stdev
            knot_points['moneyness'] = atmf_iv * knot_points_stdev * np.sqrt(tau) + 1.0
            knot_points['log_moneyness'] = np.log(knot_points.moneyness)
            knot_points['strikes'] = knot_points.moneyness * group.forward_prices.iloc[0]

            # Make sure knot points are between strike range Todo: make sure knots also satisfy all conditions
            min_max_padding = 3
            min_log_moneyness = log_moneyness.iloc[min_max_padding-1]
            max_log_moneyness = log_moneyness.iloc[-min_max_padding]

            used_knot_points = knot_points[
                (knot_points.log_moneyness >= min_log_moneyness) & (knot_points.log_moneyness <= max_log_moneyness)
            ]

            spline = LSQUnivariateSpline(log_moneyness, mid_iv, t=list(used_knot_points.log_moneyness)[1:-1], w=weightings)

            residual = spline.get_residual()
            coefficients = spline.get_coeffs()

            expiration_chain = chain[chain.expiration_dates == expiration]

            self.splines[expiration] = {
                'spline': spline,
                'knot_moneyness': knot_points,
                'used_knot_points': used_knot_points,
                'filtered_chain': group,
                'chain': expiration_chain,
                'strikes': group.strikes,
                'residual': residual,
                'coefficients': coefficients
            }

    def theos(self, strikes):
        temp_dfs = []

        for name, expiration_group in self.splines.items():
            forward_price = expiration_group['filtered_chain']['forward_prices'].iloc[0]

            moneyness = np.array(strikes)/forward_price
            log_moneyness = np.log(moneyness)

            temp_dfs.append(pd.DataFrame({
                'expiration_dates': expiration_group['filtered_chain']['expiration_dates'].iloc[0],
                'moneyness': moneyness,
                'log_moneyness': log_moneyness,
                'strikes': strikes,
                'theo_ivs': expiration_group['spline'](log_moneyness)}))

        return pd.concat(temp_dfs).reset_index(drop=True)

    @lru_cache()
    def chain_theos(self):

        temp_dfs = []

        for name, expiration_group in self.splines.items():
            expiration_group['chain']['theo_ivs'] = expiration_group['spline'](expiration_group['chain'].log_moneyness)
            temp_dfs.append(expiration_group['chain'][['log_moneyness', 'expiration_dates', 'theo_ivs']])

        # Merge theos values back to the original chain dataframe

        temp_df = pd.concat(temp_dfs)
        merged_chain = pd.merge(self.chain, temp_df, on=['log_moneyness', 'expiration_dates'], how='left')

        merged_chain['theos'] = black(
            merged_chain.option_types, merged_chain.forward_prices, merged_chain.strikes, merged_chain.tau,
            merged_chain.rates, merged_chain.theo_ivs, return_as='numpy') * np.exp(-merged_chain.rates * merged_chain.tau)

        self.chain = merged_chain
        self.chain_object.chain = merged_chain

    # def error_bounds(self, resample_n=10000):  # Todo: finnish this
    #     from numpy.random import choice
    #
    #     grouped_chain = self.grouped_chain
    #
        # for name, group in grouped_chain:
        #     log_moneyness = group.log_moneyness
        #
        #     random_selections = [list(choice(group.index, size=len(group))) for _ in range(resample_n)]
        #
        # # self.chain = chain

    def graph(self, error_bounds=False, x_axis='strikes'):
        from matplotlib import pyplot as plt

        chain = self.chain
        splines = self.splines

        # Todo: Consider moving this within the loop so that the calculation can be specific for each expiration
        min_strike = chain.strikes.min() * .99
        max_strike = chain.strikes.max() * 1.01

        # strikes = arange(min_strike, max_strike, 1).tolist()
        strikes = np.linspace(min_strike, max_strike, 500)

        theo_ivs = self.theos(strikes)

        x_units = {'strikes': 'strikes',
                   'moneyness': 'moneyness',
                   'log_moneyness': 'log_moneyness'}

        unique_expirations = theo_ivs.expiration_dates.unique()

        for expiration in unique_expirations:
            expiration_chain = chain[chain.expiration_dates == expiration]
            expiration_theo_ivs_df = theo_ivs[theo_ivs.expiration_dates == expiration]

            expiration_strikes = expiration_theo_ivs_df[x_units[x_axis]]
            expiration_theo_ivs = expiration_theo_ivs_df.theo_ivs

            expiration_chain_strikes = expiration_chain[x_units[x_axis]]

            plt.plot(expiration_strikes, expiration_theo_ivs, linewidth=.5, color='orange')
            plt.scatter(expiration_chain_strikes, expiration_chain.bid_iv, s=.5, color='red')
            plt.scatter(expiration_chain_strikes, expiration_chain.ask_iv, s=.5, color='blue')
            plt.scatter(expiration_chain_strikes, expiration_chain.mid_iv, s=.5, color='grey')

            # Todo: Make it so that it makes the closest to ATM red
            knot_points_df = splines[expiration]['used_knot_points']
            # atm_knot_point = splines[expiration]
            knot_points = knot_points_df[x_units[x_axis]]
            knot_points.apply(lambda x: plt.axvline(x=x, color='grey', linewidth=0.5, alpha=0.5))

        plt.show()

        self.chain = chain

class SVI:  # Todo: Change this to the correct format
    def __init__(self, chain_object: Chain):

        self.chain_object = chain_object

    def get_svi_model_ivs(self):
        chain = self.chain_object.chain
        chain = chain[chain['filter'] == False]
        chain = chain.sort_values(['tau', 'strikes']).reset_index(drop=True)

        from warnings import filterwarnings
        filterwarnings("ignore")

        def svi_raw(k, param):
            a = param[0]
            b = param[1]
            m = param[2]
            rho = param[3]
            sigma = param[4]
            total_variance = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
            return total_variance

        def target_function(x, strikes, forward, total_implied_variance):
            value = 0
            strikes = list(strikes)
            total_implied_variance = list(total_implied_variance)
            for i in range(len(strikes)):
                model_total_implied_variance = svi_raw(np.log(strikes[i] / forward), x)
                value = value + (total_implied_variance[i] - model_total_implied_variance) ** 2
            return value ** 0.5

        def calibrate_svi(total_impl_variance, strikes, forward, tau, initial_guess=None):
            from scipy import optimize

            forward = forward.iloc[0]
            log_moneyness = np.log(strikes / forward)
            bound = [(1e-5, max(total_impl_variance)), (1e-3, 0.99), (min(strikes), max(strikes)), (-0.99, 0.99),
                     (1e-3, 0.99)]
            if initial_guess is None:
                x0 = [0, 0, 0, 0, 0]
            else:
                x0 = initial_guess
            result = optimize.minimize(fun=target_function, x0=x0, bounds=bound, tol=1e-8, method="BFGS",
                                       args=(strikes, forward, total_impl_variance))

            # result = optimize.minimize(fun=targetfunction, bounds=bound, tol=1e-8, method="BFGS",
            #                            args=(strikes,forward,total_impl_variance))

            x = result.x
            return (svi_raw(log_moneyness, x) / tau.iloc[0]) ** .5  # ,x

        chain['tot_imp_var'] = chain['tau'] * chain['mid_iv'] ** 2
        model_ivs = chain.groupby('tau').apply(
            lambda x: calibrate_svi(x['tot_imp_var'], x['strikes'], x['forward_prices'], x['tau'])).reset_index(
            drop=True)

        if len(chain['tau'].unique()) == 1:
            from pandas import melt
            model_ivs = melt(model_ivs)
            chain['theo_ivs'] = model_ivs['value']

            chain['theos'] = black(
                chain.option_types, chain.forward_prices, chain.strikes, chain.tau, chain.rates, chain.theo_ivs,
                return_as='numpy') * np.exp(-chain.rates * chain.tau)

        else:
            chain['theo_ivs'] = model_ivs

            chain['theos'] = black(
                chain.option_types, chain.forward_prices, chain.strikes, chain.tau, chain.rates, chain.theo_ivs,
                return_as='numpy') * np.exp(-chain.rates * chain.tau)

        self.chain_object.chain = chain

        return self




class UnivariateMP:



    def calibrate_model(self,x,fit_ivs,knots,weightings=1):
        spline = LSQUnivariateSpline(x, fit_ivs, t=knots[1:-1], w=weightings)
        return spline

    def evaluate_model(self,model,x):
        return model(x)

    def evaluate_model_for_chain(self,chain_obj,calibration_info):
        from py_vollib_vectorized import vectorized_black
        from copy import deepcopy
        chain_obj = deepcopy(chain_obj)
        calibration_dict = calibration_info.copy()

        # Group both dataframes by 'expiration_dates'
        grouped_chain = chain_obj.chain.groupby(['ticker','expiration_dates'])
        results = []
        # Iterate over groups based on expiration dates
        for group, chain_group in grouped_chain:

            ticker,expiration_date = group
            # Get the corresponding calibration model (assuming one model per expiration_date)
            calibration_group = calibration_dict[ticker][expiration_date]


            # Assume 'spline' column contains spline objects ready for evaluation
            spline = calibration_group['spline']
            x_column = calibration_group['x_col']
            chain_group['theo_ivs'] = spline(chain_group[x_column])
            chain_group['theos'] = vectorized_black(chain_group['option_types'], chain_group['forward_prices'],
                                                   chain_group['strikes'], chain_group['tau'], chain_group['rates'],
                                                   chain_group['theo_ivs'], return_as='numpy') * np.exp(
                -chain_group['rates'] * chain_group['tau'])

            # Append the modified group to results
            results.append(chain_group)

        # Concatenate all modified groups into a single DataFrame
        updated_chain = pd.concat(results)

        # Update the chain_obj with the new DataFrame containing 'theo_ivs'
        chain_obj.chain = updated_chain

        return chain_obj

    # Usage in your function, assuming you can convert spline objects into a numba-compatible form




    def calibrate_model_params_to_chain(self, chain_obj, weights_col=None, knots=None, x_column='log_moneyness',
                                        y_column='mid_iv',return_chain=False):



        chain_obj = deepcopy(chain_obj)
        #TODO: add forward prices to calibration_df to incorporate spot vol model
        chain_obj.calculate_needed_col([x_column, y_column])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)

        # Validate and prepare knots data
        if isinstance(knots, int):
            knots_dict = self.compute_knot_points_for_chain(chain_obj, knots,x_column,y_column)
        elif isinstance(knots, dict):
            knots_dict = knots
        calibration_results = []
        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        for (ticker, expiration_date), group in grouped:
            current_knots = knots_dict[ticker][expiration_date]['knots']
            x_column = knots_dict[ticker][expiration_date]['x_col']
            y_column = knots_dict[ticker][expiration_date]['y_col']

            if 'filter' in group.columns:
                group = group[group['filter'] == False]


            x = group[x_column]
            fit_ivs = group[y_column]


            if weights_col:
                weights = group[weights_col]
            else:
                weights = np.ones(len(group))



            spline = self.calibrate_model(x, fit_ivs, current_knots, weightings=weights)


            knots_dict[ticker][expiration_date]['spline'] = spline
            knots_dict[ticker][expiration_date]['weights_col'] = weights_col



        calibration_dict = knots_dict.copy()
        if return_chain:
            return calibration_dict,chain_obj
        else:
            return calibration_dict



    def compute_knot_points_for_chain(self,chain_obj:Chain,points,x_column='log_moneyness',y_column='mid_iv'):

        knots_dict_ = {}
        def fill_dict(sing_expo,knots_dict,points_,x_col_,y_col_):
            expo = sing_expo['expiration_dates'].iloc[0]
            ticker = sing_expo['ticker'].iloc[0]
            if ticker not in knots_dict.keys():
                knots_dict[ticker] = {}
            knots_dict[ticker][expo] = {}
            knots_dict[ticker][expo]['knots'] = self.compute_knot_points_for_expiration(sing_expo,points_,x_col_,y_col_)
            knots_dict[ticker][expo]['x_col'] = x_column
            knots_dict[ticker][expo]['y_col'] = y_column

            return knots_dict

        chainz = chain_obj.chain.groupby(['ticker','expiration_dates'])

        for id,group in chainz:
            knots_dict_ = fill_dict(group,knots_dict_,points,x_column,y_column)

        return knots_dict_

    def calib_df_to_knots_df(self,calib_df):
        pass

    def compute_knot_points_for_expiration(self,group, points, x_column='log_moneyness', y_column='mid_iv'):
        """
        Compute knot points for a given expiration slice of option chain data.
        Assumes x_column is log moneyness and y_column is mid implied volatility by default.
        This function is prepared to be extended to handle other types of x columns.

        Parameters:
        - group: DataFrame containing data for a single expiration.
        - points: An integer key to select standard deviation arrays.
        - x_column (str): Name of the column to use as the x-axis (default 'log_moneyness').
        - y_column (str): Name of the column representing mid implied volatility (default 'mid_iv').

        Returns:
        - pd.Series: Series of knot points that are within the domain of x_column.
        """

        # Standard deviation points mapped from a key
        stdev_points = {
            13: np.array([-10, -5, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1.5, 2, 3]),
            11: np.array([-4, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 4]),
            9: np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
            7: np.array([-3, -2, -1, 0, 1, 2, 3]),
            5: np.array([-2, -1, 0, 1, 2])
        }



        # Fetch appropriate standard deviations based on 'points'
        if points not in stdev_points:
            raise ValueError(
                f'{points} is not a valid points parameter. Valid parameters are: {list(stdev_points.keys())}')

        knot_points_stdev = stdev_points[points]

        # Finding the ATM index and associated IV
        closest_to_atm_idx = (group[x_column] - 0).abs().idxmin()
        atmf_iv = group[y_column][closest_to_atm_idx]
        tau = group['tau'][closest_to_atm_idx]  # Assumes 'tau' is present in the group

        # Compute moneyness based on stdev and ATM IV
        moneyness = atmf_iv * knot_points_stdev * np.sqrt(tau) + 1.0
        log_moneyness = np.log(moneyness)

        # Filter for valid log moneyness within the range of existing x_column values
        valid_log_moneyness = log_moneyness[
            (log_moneyness >= group[x_column].min()) & (log_moneyness <= group[x_column].max())]

        return valid_log_moneyness

    def calculate_skew(self,ticker,expo,calibration_dict,point):

        return calibration_dict[ticker][expo]['spline'].derivative(n=1)(float(point))



class SSVI:
    def __init__(self):
        pass
    def ssvi(self,theta, phi, rho, k):
        """
        SSVI volatility function.

        :param theta: At-the-money variance.
        :param phi: Controls the skew.
        :param rho: Correlation between the Brownian motions.
        :param k: Log-moneyness, which is log(strike / forward price).
        :return: Implied volatility for the given parameters and log-moneyness.
        """
        term1 = theta / 2
        term2 = (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))
        return term1 * term2


    def objective_function(self,params, k, market_vols):
        """
        Objective function to minimize: RMSE between market and model implied volatilities.

        :param params: Array of parameters [theta, phi, rho].
        :param k: Array of log-moneyness values.
        :param market_vols: Array of market implied volatilities.
        :return: Root mean squared error.
        """
        theta, phi, rho = params
        model_vols = self.ssvi(theta, phi, rho, k)
        return np.sqrt(np.mean((market_vols - model_vols) ** 2))

    def objective_function_gatheral(self,params,k,market_vols_mid,market_vols_bid,market_vols_ask):
        theta, phi, rho = params
        model_vols = self.ssvi(theta, phi, rho, k)
        return np.sqrt(np.mean((((market_vols_mid - model_vols) ** 2) / (market_vols_ask - market_vols_bid)**2)) )
        #weighter = 1/(np.exp((market_vols_ask-market_vols_bid)))
        #return np.sqrt(np.mean((((market_vols_mid - model_vols) ** 2)*weighter)))

    def objective_func_new(self,params,k,market_vols_mid,market_vols_bid,market_vols_ask):
        theta, phi, rho = params
        model_vols = self.ssvi(theta, phi, rho, k)
        weights = 1/(market_vols_ask - market_vols_bid)
        obj_1 = np.mean(  ((market_vols_mid - model_vols)*weights)**2   )
        lambda_ = 2
        penalty_ba = np.maximum(0, market_vols_bid - model_vols) + np.maximum(0, model_vols - market_vols_ask)

        obj_2 = lambda_ * np.mean((penalty_ba))
        #print(obj_2 + obj_1)
        return obj_2 + obj_1

    def objective_func_gathel(self, params, k, market_vols_mid, market_vols_bid, market_vols_ask):
        theta, phi, rho = params
        model_vols = self.ssvi(theta, phi, rho, k)
        weights = 1 / (market_vols_ask - market_vols_bid)
        return np.mean( ((market_vols_mid - model_vols)*weights)/market_vols_mid  )



        #return np.sqrt(np.mean((((market_vols_mid - model_vols) ** 2) / (market_vols_ask - market_vols_bid)**2)) )



    def fit_ssvi(self,strikes, forward, market_vols,market_vols_bid=[],market_vols_ask=[],initial_params=[]):
        """
        Fits the SSVI model to market data.

        :param strikes: Array of strike prices.
        :param forward: Forward price of the underlying.
        :param market_vols: Array of market implied volatilities.
        :return: Optimized parameters and the success status of the optimization.
        """
        # Compute log-moneyness
        k = np.log(strikes / forward)

        # Initial guesses for theta, phi, rho
        if len(initial_params) > 0:
            initial_params = initial_params
        else:

            initial_params = np.array([0.2, 0.1, 0.0])  # Example initial values; adjust based on typical market data

        # Bounds for theta, phi, rho
        bounds = [(0.001, 5), (0.001, 1000), (-0.999, 0.999)]

        # Minimize the objective function
        if len(market_vols_ask) > 0:
            result = minimize(self.objective_function_gatheral, initial_params, args=(k, market_vols,market_vols_bid,market_vols_ask), bounds=bounds)
        else:
            result = minimize(self.objective_function, initial_params,
                              args=(k, market_vols), bounds=bounds,tol=1e-10)


        return result.x, result.success


    def evaluate_model_for_chain(self, chain_obj, params_dict):
        from py_vollib_vectorized import vectorized_black
        grouped_chain = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        results = []

        for group, chain_group in grouped_chain:

            ticker, expiration_date = group
            calibration_group = params_dict[ticker][expiration_date]['params']
            try:
                chain_group['theo_ivs'] = self.ssvi_vols(chain_group['strikes'], chain_group['forward_prices'],
                                                      calibration_group)
            except IndexError:
                #print(calibration_group)
                #import traceback
                pass
                #traceback.print_exc()

            chain_group['theos'] = vectorized_black(chain_group['option_types'], chain_group['forward_prices'],
                                                    chain_group['strikes'], chain_group['tau'], chain_group['rates'],
                                                    chain_group['theo_ivs'], return_as='numpy') * np.exp(
                -chain_group['rates'] * chain_group['tau'])
            results.append(chain_group)

        chain_obj.chain = pd.concat(results)
        return chain_obj


    def calibrate_model_params_to_chain(self, chain_obj, x_col='strikes', y_col='mid_iv', return_chain=False,
                                        obj_func='gatheral', initial_guess=[], url=None,recalibrate_method='standard'):

        #recalibrate_method is either 'standard' or 'kalman'

        if url:
            return self.calibrate_model_params_to_chain_from_url(chain_obj, x_col=x_col, y_col=y_col, obj_func=obj_func,
                                                                 initial_guess=initial_guess, url=url)

        from copy import deepcopy

        if initial_guess != []:

            return self.recalibrate_model(chain_obj,initial_guess)
        chain_obj = deepcopy(chain_obj)
        if 'filter' in chain_obj.chain.columns:
            chain_obj.chain = chain_obj.chain[chain_obj.chain['filter'] == False]

        chain_obj.calculate_needed_col([x_col, y_col])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)


        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])

        params_dict = {}

        lst = [i for i in chain_obj.chain.groupby(['ticker', 'expiration_dates'])]

        for (ticker, expiration_date), group in grouped:


            x = group[x_col]
            y = group[y_col]


            if obj_func != 'gatheral':
                params, success = self.fit_ssvi(x, group['forward_prices'], y)
            else:
                params, success = self.fit_ssvi(x, group['forward_prices'], y, group['bid_iv'], group['ask_iv'])

            if ticker not in params_dict.keys():
                params_dict[ticker] = {}

            params_dict[ticker][expiration_date] = {}
            params_dict[ticker][expiration_date]['params'] = params

            params_dict[ticker][expiration_date]['x_col'] = x_col
            params_dict[ticker][expiration_date]['y_col'] = y_col
            params_dict[ticker][expiration_date]['obj_func'] = obj_func
            params_dict[ticker][expiration_date]['recalibrate_method'] = recalibrate_method
            if recalibrate_method == 'kalman':
                ukf,kalman_init_args = self.initialize_kalman_filter(group,params)

                params_dict[ticker][expiration_date]['kalman_obj'] = ukf
                params_dict[ticker][expiration_date]['kalman_init_args'] = kalman_init_args

            else:
                params_dict[ticker][expiration_date]['kalman_obj'] = None
                params_dict[ticker][expiration_date]['kalman_init_args'] = None



        return params_dict


    def recalibrate_model(self,chain_obj, params_dict):
        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        new_dict = {}

        for (ticker, expiration_date), group in grouped:
            if ticker not in new_dict.keys():
                new_dict[ticker] = {}
            params_dict_expo = params_dict[ticker][expiration_date]



            if params_dict_expo['recalibrate_method'] == 'standard':

                group = group[group['filter'] == False]
                new_params,_ = self.fit_ssvi(group[params_dict_expo['x_col']],group['forward_prices'],group[params_dict_expo['y_col']],group['bid_iv'],group['ask_iv'],params_dict_expo['params'])


                new_dict[ticker][expiration_date] = params_dict_expo
                new_dict[ticker][expiration_date]['params'] = new_params
            elif params_dict_expo['recalibrate_method'] == 'kalman':
                new_dict[ticker][expiration_date] = self.recalibrate_model_using_ukf(group,params_dict_expo)
        return new_dict

    def recalibrate_model_using_ukf(self, chain_sing_exp, inner_params_dict):
        y_col, x_col = inner_params_dict['y_col'], inner_params_dict['x_col']
        strikes = chain_sing_exp[x_col]
        forward = chain_sing_exp['forward_prices'].iloc[0]

        ukf = inner_params_dict['kalman_obj']
        kalman_init_args = inner_params_dict['kalman_init_args']

        #logging.debug("Before predict:")
        #logging.debug(f"UKF state: {ukf.x}")
        ukf.predict()
        x_pred = ukf.x
        #logging.debug(f"UKF state after predict: {x_pred}")

        pred_ssvi = self.hx(x_pred, strikes, forward)
        #logging.debug(f"pred_ssvi: {pred_ssvi}")
        len_cross = len(
            chain_sing_exp[(chain_sing_exp['bid_iv'] > pred_ssvi) | (chain_sing_exp['ask_iv'] < pred_ssvi)])
        #logging.debug(f"len_cross: {len_cross}")
        if len_cross > 3:
            ukf.Q *= 3
            #logging.debug(f"UKF Q after scaling: {ukf.Q}")
            ukf.predict()
            x_pred = ukf.x
            #logging.debug(f"UKF state after second predict: {x_pred}")

        high_variance = 1e6  # Adjust as needed
        min_variance = 1e-10
        R_values = ((chain_sing_exp['ask_iv'] - chain_sing_exp['bid_iv']) / 3) ** 2

        # Handle NaNs in R_values
        if np.any(np.isnan(R_values)):
            pass
            #logging.error("R_values contain NaNs. Replacing NaNs with a default high variance.")
            #R_values = np.nan_to_num(R_values, nan=1e6)  # Replace NaNs with high variance

        # Ensure R_values are positive and replace filtered entries with high variance
        R_values = np.where(chain_sing_exp['filter'] == False, R_values, high_variance)
        R_values = np.maximum(R_values, min_variance)

        #logging.debug(f"R_values before assignment: {R_values}")
        ukf.R = np.diag(R_values)
        #logging.debug(f"UKF R matrix: {ukf.R}")

        z = np.array(chain_sing_exp[y_col])
        #logging.debug(f"Measurement vector z: {z}")

        ukf.update(z, strikes=strikes, forwards=forward)
        #logging.debug(f"UKF state after update: {ukf.x}")
        #logging.debug(f"UKF P matrix after update: {ukf.P}")

        # Check for NaNs after update
        if np.any(np.isnan(ukf.x)):
            #logging.error("UKF state vector contains NaNs after update. Reinitializing state.")
            return  # Handle accordingly
        if np.any(np.isnan(ukf.P)):
            #logging.error("UKF P matrix contains NaNs after update. Reinitializing covariance.")
            return  # Handle accordingly

        # Continue with Q estimation if adaptive Q is enabled
        if 'P_j_hist' in kalman_init_args['adaptive_Q_info'].keys():
            delta_x_hist, P_j_hist, sigma_cov_hist = (
                kalman_init_args['adaptive_Q_info']['delta_x_hist'],
                kalman_init_args['adaptive_Q_info']['P_j_hist'],
                kalman_init_args['adaptive_Q_info']['sigma_cov_hist']
            )
            sigmas_pred = ukf.sigmas_f.copy()
            Delta_x_j = (ukf.x_post - x_pred)

            if Delta_x_j.ndim == 1:
                Delta_x_j = Delta_x_j.reshape(-1,1)
            #print('dxj')
            #print(Delta_x_j)


            P_j = ukf.P.copy()  # (3,3)

            sigma_cov = np.zeros_like(P_j)
            for i, w in enumerate(ukf.Wc):
                diff = (sigmas_pred[i] - x_pred).reshape(-1, 1)
                if np.isnan(w) or np.any(np.isnan(diff)):
                    #logging.error(f"NaN detected in weight or diff at index {i}. Skipping this sigma.")
                    continue  # Skip this sigma
                sigma_cov += w * (diff @ diff.T)

            #logging.debug(f"sigma_cov before checking: {sigma_cov}")
            if np.any(np.isnan(sigma_cov)):
                #logging.error("sigma_cov contains NaNs after calculation. Skipping Q estimation.")
                return  # Handle accordingly

            delta_x_hist.append(Delta_x_j)
            P_j_hist.append(P_j)
            sigma_cov_hist.append(sigma_cov)
            if len(sigma_cov_hist) > 360:
                del sigma_cov_hist[0]
                del P_j_hist[0]
                del delta_x_hist[0]

            try:
                Q_ = estimate_Q_from_eq37(delta_x_hist, P_j_hist, sigma_cov_hist)
                ukf.Q = Q_
            except Exception as e:
                print('UKF Q not currently updating')
            #except Exception as e:
            #    import traceback
            #    logging.error("ukf Q not currently updating")
            #    traceback.print_exc()
                #logging.error(traceback.format_exc())
                #logging.debug(f"Current Q: {ukf.Q}")
                #logging.debug(f"delta_x_hist: {delta_x_hist}")
                #logging.debug(f"P_j_hist: {P_j_hist[-1]}")
                #logging.debug(f"sigma_cov_hist: {sigma_cov_hist[-1]}")

        # Final checks before exiting
        if np.any(np.isnan(ukf.Q)):

            #logging.error("UKF Q matrix contains NaNs after update. Reverting to default Q.")
            from kalman_help import make_positive_definite
            ukf.Q = make_positive_definite(ukf.Q)  # Revert or set to default


        kalman_init_args['adaptive_Q_info']['delta_x_hist'], kalman_init_args['adaptive_Q_info']['P_j_hist'], \
        kalman_init_args['adaptive_Q_info']['sigma_cov_hist'] = delta_x_hist,P_j_hist ,sigma_cov_hist
        inner_params_dict['kalman_init_args'] = kalman_init_args

        inner_params_dict['kalman_obj'] = ukf
        inner_params_dict['params'] = self.inverse_transform(ukf.x_post)
        return inner_params_dict



    def p_transform(self,x):
            # return np.arcsin(x-1)
            return np.arcsin((2 * (x + 1) / (2)) - 1)

    def p_transform_min(self,x):
        return np.sqrt((1 + x) ** 2 - 1)

    def p_inv_transform_min(self,x):
        return -1 + np.sqrt(x ** 2 + 1)

    def transform(self,lst):
        item_1 = lst[0]
        item_2 = lst[1]

        new_1 = self.p_transform_min(item_1)
        new_2 = self.p_transform_min(item_2)
        new_3 = self.p_transform(lst[-1])

        return [new_1, new_2, new_3]

    def p_inv_transform(self,x):
        return np.sin(x)


    def inverse_transform(self,lst):
        new_1 = self.p_inv_transform_min(lst[0])
        new_2 = self.p_inv_transform_min(lst[1])
        new_3 = self.p_inv_transform(lst[2])
        return [new_1, new_2, new_3]

    def fx(self,x, dt):
        return x

    def fx_rw(self,x,dt):
        return x

    def hx(self,x, strikes, forwards):
        x = self.inverse_transform(x)
        ssvi_model_ivs = self.ssvi_vols(strikes, forwards, x)
        if not np.all(np.isfinite(ssvi_model_ivs)):
            raise ValueError("SSVI returned non-finite implied volatilities.")
        return ssvi_model_ivs

    def initialize_kalman_filter(self,chain_sing_expo,param0,kalman_init_args={'process_model': 'random_walk', 'process_model_params': None,
                            'Q_init': np.identity(3) * 1e-2,'sigma_point_args':[3,0.5,2,-1],'adaptive_Q_info':{'delta_x_hist':[],'P_j_hist':[],'sigma_cov_hist':[]}}):

        from kalman_help import make_positive_definite,estimate_Q_from_eq37
        from filterpy.kalman import MerweScaledSigmaPoints
        from filterpy.kalman import UnscentedKalmanFilter as UKF
        alpha = param0
        sigma_args = kalman_init_args['sigma_point_args']
        if kalman_init_args['process_model'] == 'random_walk':
            fx = self.fx_rw
        points = MerweScaledSigmaPoints(n=sigma_args[0], alpha=sigma_args[1], beta=sigma_args[2], kappa=sigma_args[3])
        ukf = UKF(
            dim_x=len(alpha),
            dim_z=len(chain_sing_expo['strikes']),
            fx=fx,
            hx=self.hx,
            dt=60,
            points=points
        )
        ukf.x = self.transform(alpha)
        ukf.Q = kalman_init_args['Q_init']
        ukf.P = kalman_init_args['Q_init']*3
        return ukf,kalman_init_args




    def calibrate_model_params_to_chain2(self,chain_obj,x_col='strikes',y_col='mid_iv',return_chain=False,obj_func='gatheral',initial_guess=[],url=None):
        if url:
            return self.calibrate_model_params_to_chain_from_url(chain_obj,x_col=x_col,y_col=y_col,obj_func=obj_func,initial_guess=initial_guess,url=url)
        from copy import deepcopy
        chain_obj = deepcopy(chain_obj)
        if 'filter' in chain_obj.chain.columns:
            chain_obj.chain = chain_obj.chain[chain_obj.chain['filter'] == False]
        chain_obj.calculate_needed_col([x_col, y_col])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)

        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        params_dict = {}
        for (ticker, expiration_date), group in grouped:


            x = group[x_col]
            y = group[y_col]

            if len(initial_guess)  > 0:
                initial_guess_= initial_guess[ticker][expiration_date]
            else:
                initial_guess_ = []

            if obj_func != 'gatheral':
                params,success = self.fit_ssvi(x,group['forward_prices'],y,initial_params=initial_guess_)
            else:
                params, success = self.fit_ssvi(x, group['forward_prices'], y,group['bid_iv'],group['ask_iv'],initial_params=initial_guess_)

            if ticker not in params_dict.keys():
                params_dict[ticker] = {}

            params_dict[ticker][expiration_date] = params
           # time.sleep(0.01)

        return params_dict

    def ssvi_vols(self,strikes, forward, params):
        k = np.log(strikes / forward)
        return self.ssvi(params[0], params[1], params[2], k)

    def evaluate_model_for_chain2(self, chain_obj, params_dict):
        from py_vollib_vectorized import vectorized_black
        grouped_chain = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        results = []

        for group, chain_group in grouped_chain:
            ticker, expiration_date = group
            calibration_group = params_dict[ticker][expiration_date]


            chain_group['theo_ivs'] = self.ssvi_vols(chain_group['strikes'],chain_group['forward_prices'],calibration_group)
            chain_group['theos'] = vectorized_black(chain_group['option_types'], chain_group['forward_prices'],
                                                    chain_group['strikes'], chain_group['tau'], chain_group['rates'],
                                                    chain_group['theo_ivs'], return_as='numpy') * np.exp(
                -chain_group['rates'] * chain_group['tau'])
            results.append(chain_group)

        chain_obj.chain = pd.concat(results)
        return chain_obj

    def make_calibration_result_jsonable(self,result):
        for key, value in result.items():
            for subkey, arr in value.items():
                if isinstance(arr, np.ndarray):
                    result[key][subkey] = arr.tolist()
        return result

    def remake_jsonable_calibration_result_to_original(self,result):
        new_dict = {}
        for key,value in result.items():
            new_dict[key] = {}
            for subkey,value in value.items():
                new_dict[key][pd.to_datetime(subkey)] = np.array(result[key][subkey])
        return new_dict

    def make_calibration_input_jsonable(self,payload):
        from copy import deepcopy
        init_guess = deepcopy(payload['data']['params']['initial_guess'])
        if init_guess == []:
            return payload
        else:
            new_init_guess = {}
            for ticker,value in init_guess.items():
                new_init_guess[ticker] = {}
                for expo,_ in value.items():
                    params = init_guess[ticker][expo]
                    new_init_guess[ticker][str(expo)] = list(params)

            payload['data']['params']['initial_guess'] = new_init_guess
            return payload





    def revert_calibration_input(self,data):
        return data


    def calibrate_model_params_to_chain_from_url(self,chain_obj,url,x_col='strikes',y_col='mid_iv',obj_func='gatheral',initial_guess=[]):
        import requests
        df = make_json_serializable(chain_obj.chain)

        #url = "http://0.0.0.0:8000/process_data/calibration"
        url = f"{url}/calibration"
        payload = {
            "data": {
                'chain': df.to_dict(),
                'model_obj_str': 'SSVI',
                'params': {
                    'x_col': x_col,
                    'y_col': y_col,
                    'return_chain': False,
                    'obj_func': obj_func,
                    'initial_guess': initial_guess
                }
            }
        }
        payload = self.make_calibration_input_jsonable(payload)
        response = requests.post(url, json=payload).json()
        #response = requests.post(url, json=payload)
        #print(response)
        response = self.remake_jsonable_calibration_result_to_original(response)

        return response






class ChainReinit:
    def __init__(self, chain,mid_price_method=MidPrices.mid_prices):

        input_chain = chain.copy(deep=True)
        self.input_chain = input_chain
        self.chain = chain

        self._mid_price_method = mid_price_method


    def __calculate__(self):

        self.chain = self._mid_price_method(self.chain)
        self._forward_price()
        self._filter_itm_options()
        self._implied_volatility()
        self._moneyness()
        self._log_moneyness()
        self._bsm_delta()

        return self

    def _forward_price(self):
        from numpy import exp
        chain = self.chain


        def _forward_price_inner(chain):
            # Create separate dataframes for calls and puts
            calls = chain[chain.option_types == 'c'].copy()
            puts = chain[chain.option_types == 'p'].copy()
        
            # Merge calls and puts on 'strikes', 'expiration_dates', 'tau', and 'rates'
            chain_merged = calls.merge(puts, on=['strikes', 'expiration_dates', 'tau', 'rates'],
                                       suffixes=('_call', '_put'))
        
            # Calculate mid absolute differences
            chain_merged['mid_abs_diffs'] = abs(chain_merged['mid_prices_call'] - chain_merged['mid_prices_put'])
        
            # Calculate the index of minimum differences by expiration date
            min_diff_idx = chain_merged.groupby('expiration_dates')['mid_abs_diffs'].idxmin()
        
        
            # Create a temporary dataframe with minimum differences
            min_diff_chain = chain_merged.loc[min_diff_idx]
        
            pd.set_option('display.max_columns', None)
        
            # Compute forward prices
            min_diff_chain['forward_prices'] = min_diff_chain.strikes + np.exp(
                min_diff_chain.rates * min_diff_chain.tau) * \
                                               (min_diff_chain['mid_prices_call'] - min_diff_chain['mid_prices_put'])
        
            print(min_diff_chain)
        
            # Merge the forward prices to the main chain DataFrame
            chain = chain.merge(
                min_diff_chain[['expiration_dates', 'forward_prices']], on='expiration_dates', how='left')
        
            return chain
        
        chain = chain.groupby(['ticker', 'snap_shot_dates', 'snap_shot_times']).apply(
            lambda x: _forward_price_inner(x)).reset_index(
            drop=True)
        
        self.chain = chain
        return self

        """
        def _forward_price_inner(chain):
            # Create separate dataframes for calls and puts
            calls = chain[chain.option_types == 'c'].copy()
            puts = chain[chain.option_types == 'p'].copy()

            # Merge calls and puts on 'strikes', 'expiration_dates', 'tau', and 'rates'
            chain_merged = calls.merge(puts, on=['strikes', 'expiration_dates', 'tau', 'rates'],
                                       suffixes=('_call', '_put'))

            # Calculate mid absolute differences
            chain_merged['mid_abs_diffs'] = abs(chain_merged['mid_prices_call'] - chain_merged['mid_prices_put'])

            # Find top three minimum absolute differences by expiration date
            top_min_diffs = chain_merged.groupby('expiration_dates')['mid_abs_diffs'].nsmallest(3).reset_index()

            # Filter data to only include these top three per expiration
            chain_top_min_diffs = chain_merged.loc[top_min_diffs['level_1']]

            # Calculate combined bid-ask spreads for top three minimum diffs
            chain_top_min_diffs['bid_ask_spread'] = (
                        chain_top_min_diffs['ask_prices_call'] - chain_top_min_diffs['bid_prices_call'] +
                        chain_top_min_diffs['ask_prices_put'] - chain_top_min_diffs['bid_prices_put'])

            # Select the index with the smallest bid-ask spread for each expiration
            idx_min_spread = chain_top_min_diffs.groupby('expiration_dates')['bid_ask_spread'].idxmin()

            # Create a dataframe with the best forward price candidate strikes
            min_diff_chain = chain_top_min_diffs.loc[idx_min_spread]

            # Compute forward prices
            min_diff_chain['forward_prices'] = min_diff_chain['strikes'] + np.exp(
                min_diff_chain['rates'] * min_diff_chain['tau']) * (min_diff_chain['mid_prices_call'] - min_diff_chain[
                'mid_prices_put'])

            # Merge the forward prices to the main chain DataFrame
            chain = chain.merge(min_diff_chain[['expiration_dates', 'forward_prices']], on='expiration_dates',
                                how='left')

            return chain
        """

        # Usage within the class or script
        chain = chain.groupby(['ticker', 'snap_shot_dates', 'snap_shot_times']).apply(
            lambda x: _forward_price_inner(x)).reset_index(drop=True)

        self.chain = chain
        return self

    def _filter_itm_options(self):
        from pandas import concat

        if 'mid_prices_call' not in self.chain.columns:
            self.chain = self._mid_price_method(self.chain)

        if 'forward_prices' not in self.chain.columns:
            self._forward_price()

        chain = self.chain

        calls = chain[chain.option_types == 'c'].copy()
        puts = chain[chain.option_types == 'p'].copy()

        calls = calls[calls.strikes >= calls.forward_prices]
        puts = puts[puts.strikes < puts.forward_prices]

        chain_otmf = concat([calls, puts]).sort_values(['tau', 'strikes']).reset_index(drop=True)

        self.chain = chain_otmf

    def _implied_volatility(self):
        from numba import NumbaDeprecationWarning
        from warnings import filterwarnings
        from numpy import errstate
        from numpy import nan

        if 'mid_prices_call' not in self.chain.columns:
            self.chain = self._mid_price_method(self.chain)

        if 'forward_prices' not in self.chain.columns:
            self._forward_price()

        filterwarnings('ignore', category=NumbaDeprecationWarning)  # There is a problem with the vec vollib library

        from py_vollib_vectorized import vectorized_implied_volatility_black as iv_black

        chain = self.chain

        # Todo: Turn this all into a singe iv_black function call

        try:
            with errstate(divide='raise', invalid='ignore'):

                chain['bid_iv'] = iv_black(chain.bid_prices,
                                           chain.forward_prices,
                                           chain.strikes,
                                           chain.rates,
                                           chain.tau,
                                           chain.option_types,
                                           return_as='array')
        except ZeroDivisionError:
            chain['bid_iv'] = nan

        try:
            with errstate(divide='raise', invalid='ignore'):

                chain['ask_iv'] = iv_black(chain.ask_prices,
                                           chain.forward_prices,
                                           chain.strikes,
                                           chain.rates,
                                           chain.tau,
                                           chain.option_types,
                                           return_as='array')
        except ZeroDivisionError:

            chain['ask_iv'] = nan

        try:
            with errstate(divide='raise', invalid='ignore'):

                # chain['mid_iv'] = iv_black(chain.mid_prices,
                #                           chain.forward_prices,
                #                           chain.strikes,
                #                           chain.rates,
                #                           chain.tau,
                #                           chain.option_types,
                #                            return_as='array')

                chain['mid_iv'] = iv_black(chain['mid_prices'], chain['forward_prices'], chain['strikes'],
                                           chain['rates'], chain['tau'], chain['option_types'], return_as='array')

        except ZeroDivisionError:
            chain['mid_iv'] = nan

        chain = chain.sort_values(['tau', 'strikes'])

        self.chain = chain
        return self

    def _moneyness(self):
        chain = self.chain

        chain['moneyness'] = chain.strikes / chain.forward_prices

        self.chain = chain

    def _log_moneyness(self):  # Todo: Consider calculating this based of moneyness()
        from numpy import log
        chain = self.chain

        chain['moneyness'] = chain['strikes']/chain['forward_prices']
        chain['log_moneyness'] = log(chain.moneyness)

        self.chain = chain

    def _bsm_delta(self):
        from py_vollib_vectorized import vectorized_delta

        if 'mid_iv' not in self.chain.columns:
            self._implied_volatility()

        chain = self.chain

        chain['delta'] = vectorized_delta(chain.option_types, chain.forward_prices, chain.strikes, chain.tau,
                                          chain.rates, chain.mid_iv)
        self.chain = chain
        return self

    def delta_buckets(self):  # Todo: Make the delta buckets a param so they can be changed
        from pandas import cut

        chain = self.chain
        bins = [-float('inf'), -50, -0.3, -0.1, 0, 0.1, 0.3, 0.5, float('inf')]
        labels = ['-inf to -50', '-.30--.50', '-.30--.10', '-.10-0', '0-.10', '.10-.30', '.30-.50', '50+']
        chain['delta_bucket'] = cut(chain.delta, bins=bins, labels=labels, right=False)
        self.chain = chain

    def filter_dataframe(self, column, operator, value):
        chain = self.chain

        operators = {
            '==': chain[column] == value,
            '>': chain[column] > value,
            '>=': chain[column] >= value,
            '<': chain[column] < value,
            '<=': chain[column] <= value,
            '!=': chain[column] != value,
        }

        try:
            chain = chain[operators[operator]].reset_index(drop=True)
            self.chain = chain

        except KeyError:
            valid_operators = ", ".join(operators.keys())
            raise ValueError(f"Invalid operator '{operator}'. Valid operators are: {valid_operators}")

    def arbitrage_free_mid_vols(self, by_expiry=True):
        from arbitrage_repair import constraints, repair
        #from arbitragerepair import constraints,repair
        from py_vollib_vectorized import vectorized_implied_volatility, vectorized_black
        chain = self.chain.sort_values(['ticker', 'snap_shot_dates', 'expiration_dates', 'strikes']).reset_index(
            drop=True)

        def get_arb_free_mid_for_tick(chain_tick):
            '''

            import time
            s = time.time()


            puts = chain_tick[chain_tick['option_types'] == 'p']
            calls = chain_tick[chain_tick['option_types'] == 'c']

            # Calculate equivalent call prices from put prices using put-call parity
            # C = P + S - K * exp(-rT)
            calculated_call_mids = puts['mid_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
               -puts['rates'] * puts['tau'])
            calculated_call_bids = puts['bid_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
                -puts['rates'] * puts['tau'])
            #calculated_call_asks = puts['ask_prices'] + puts['forward_prices'] - puts['strikes'] * np.exp(
                -puts['rates'] * puts['tau'])

            # Create a new DataFrame for the calculated call prices from puts
            derived_calls = pd.DataFrame({
                'strikes': puts['strikes'],
                'mid_prices': calculated_call_mids,
                'bid_prices': calculated_call_bids,
                'ask_prices': calculated_call_asks,
                'forward_prices': puts['forward_prices'],
                'rates': puts['rates'],
                'tau': puts['tau'],
                'ticker': puts['ticker']
            }, columns=['strikes', 'mid_prices', 'bid_prices', 'ask_prices', 'forward_prices', 'rates', 'tau',
                        'ticker'])

            # Concatenate derived call prices with existing call options
            combined_calls = pd.concat([calls, derived_calls], ignore_index=True)

            # Now you have combined call options where 'combined_calls' includes both existing and derived call prices
            call_mids = combined_calls['mid_prices']
            call_bids = combined_calls['bid_prices']
            call_asks = combined_calls['ask_prices']
            '''

            call_mids = vectorized_black(flag='c',
                                         F=chain_tick['forward_prices'],
                                         t=chain_tick['tau'],
                                         K=chain_tick['strikes'],
                                         r=chain_tick['rates'],
                                         sigma=chain_tick['mid_iv'],return_as='array')

            call_bids = vectorized_black(flag='c',
                                         F=chain_tick['forward_prices'],
                                         t=chain_tick['tau'],
                                         K=chain_tick['strikes'],
                                         r=chain_tick['rates'],
                                         sigma=chain_tick['bid_iv'],return_as='array')

            call_asks = vectorized_black(flag='c', F=chain_tick['forward_prices'], t=chain_tick['tau'],
                                         K=chain_tick['strikes'], r=chain_tick['rates'], sigma=chain_tick['ask_iv'],return_as='array')

            normaliser = constraints.Normalise()
            normaliser.fit(chain_tick['tau'].to_numpy(), chain_tick['strikes'].to_numpy(), call_mids,
                           chain['forward_prices'].to_numpy())

            T1, K1, C1 = normaliser.transform(chain['tau'].to_numpy(), chain['strikes'].to_numpy(),
                                              call_mids)

            mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)

            bid_spread = call_mids - call_bids
            ask_spread = call_asks - call_mids
            spreads = np.array([ask_spread, bid_spread])

            epsilon = repair.l1ba(mat_A, vec_b, C1, spreads)

            K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon)

            call_mids_arb_free = C0
            #chain_tick['strikes'] = K0

            chain_tick['mid_iv_arb_free'] = vectorized_implied_volatility(flag='c',
                                                                                S=chain_tick['forward_prices'],
                                                                                t=chain_tick['tau'],
                                                                                K=chain_tick['strikes'],
                                                                                r=0,
                                                                                price=call_mids_arb_free)

            return chain_tick

        if by_expiry:
            grouper = ['ticker', 'snap_shot_datetimes', 'expiration_dates']
        else:
            grouper = ['snap_shot_datetimes', 'ticker']

        #import time
        #s = time.time()
        chain = chain.groupby(grouper).apply(lambda x: get_arb_free_mid_for_tick(x)).reset_index(drop=True)
        #print(time.time() - s)

        self.chain = chain


    def calculate_needed_col(self, needed_cols, args_dict={}, kwargs_dict={}):
        '''
        example_usage:

       needed_columns = ['mid_iv', 'log_moneyness']

       args_dict = {
           'mid_iv': (['input for arg1'],),  # Correct arguments
       }

       kwargs_dict = {
           'mid_iv': {'example_kwarg': 'keyword_argument for iv'}
       }

       '''

        from inspect import signature

        required_methods = {
            'mid_iv': self._implied_volatility,
            'bid_iv': self._implied_volatility,
            'ask_iv': self._implied_volatility,
            'log_moneyness': self._log_moneyness,
            'delta_bucket': self.delta_buckets,
            'delta': self._bsm_delta,
            'moneyness': self._moneyness,
            'forward_prices': self._forward_price,
            'mid_iv_arb_free':self.arbitrage_free_mid_vols
        }

        for col in needed_cols:
            if col not in self.chain.columns:
                method = required_methods[col]
                method_args = args_dict.get(col, ())
                method_kwargs = kwargs_dict.get(col, {})
                sig = signature(method)
                parameters = sig.parameters

                # Check for missing required arguments
                missing_args = [p for p in parameters.values()
                                if p.default == p.empty and  # Check if parameter is required
                                p.name not in method_kwargs and  # Not provided as a keyword argument
                                len(method_args) < list(parameters.keys()).index(
                        p.name) + 1]  # Not provided as a positional argument

                if missing_args:
                    missing_params = ', '.join(p.name for p in missing_args)
                    raise ValueError(
                        f"Missing required arguments for method `{method.__name__}` when calculating '{col}': {missing_params}")

                method(*method_args, **method_kwargs)




class PolynomialActivation:
    def __init__(self):
        pass

    def model_volatility(self, k, a0, a1, a2, a_plus, a_minus):
        """Calculate the model volatility."""
        asymm_sum = sum(
            a_plus[i] * (k ** (i + 3)) * (k >= 0) + a_minus[i] * (k ** (i + 3)) * (k < 0) for i in range(len(a_plus)))
        return a0 + a1 * k + a2 * k ** 2 + asymm_sum

    def objective_function(self, params, k, mid_vol, bid_iv, ask_iv):
        """Objective function to minimize."""
        n = (len(params) - 3) // 2 + 3  # Calculate n based on the number of parameters
        a0 = params[0]
        a1 = params[1]
        a2 = params[2]
        a_plus = params[3:3 + n - 3]
        a_minus = params[3 + n - 3:]
        model_vol = self.model_volatility(k, a0, a1, a2, a_plus, a_minus)
        weights = 1 / (ask_iv - bid_iv) ** 2
        return np.sum(weights * (model_vol - mid_vol) ** 2)

    def calibrate_model(self, k, mid_vol, bid_iv, ask_iv, n, initial_guess=None):
        """Calibrate the model to the given data."""
        if initial_guess is None:
            initial_guess = np.array([0.1] * (3 + 2 * (n - 3)))  # Example initial guess for a0, a1, a2, a_plus, and a_minus
        bounds = [(None, None)] * len(initial_guess)  # No bounds for parameters
        result = minimize(self.objective_function, initial_guess,
                          args=(k, mid_vol, bid_iv, ask_iv),
                          bounds=bounds)
        return result.x

    def get_model_values(self, k, params):
        """Get model values for the given k using the calibrated parameters."""
        n = (len(params) - 3) // 2 + 3  # Calculate n based on the number of parameters
        a0 = params[0]
        a1 = params[1]
        a2 = params[2]
        a_plus = params[3:3 + n - 3]
        a_minus = params[3 + n - 3:]
        return self.model_volatility(k, a0, a1, a2, a_plus, a_minus)

    def calibrate_model_params_to_chain(self, chain_obj, y_column='mid_iv', n=4, initial_guess=None):
        """Calibrate model parameters for each expiration in the chain."""
        chain_obj = deepcopy(chain_obj)
        chain_obj.calculate_needed_col([y_column, 'log_moneyness'])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)

        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        params_dict = {}
        for (ticker, expiration_date), group in grouped:
            if 'filter' in group.columns:
                group = group[group['filter'] == False]

            k = group['log_moneyness'].values
            mid_vol = group[y_column].values
            bid_iv = group['bid_iv'].values
            ask_iv = group['ask_iv'].values

            if initial_guess:
                initial_guess_ = initial_guess.get(ticker, {}).get(expiration_date, None)
            else:
                initial_guess_ = None


            params = self.calibrate_model(k, mid_vol, bid_iv, ask_iv, n, initial_guess_)
            if ticker not in params_dict:
                params_dict[ticker] = {}
            params_dict[ticker][expiration_date] = params

        return params_dict

    def evaluate_model_for_chain(self, chain_obj, params_dict):
        """Evaluate the model for each expiration in the chain and add 'theo_ivs'."""
        from py_vollib_vectorized import vectorized_black

        grouped_chain = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        results = []

        for group, chain_group in grouped_chain:
            ticker, expiration_date = group
            calibration_group = params_dict[ticker][expiration_date]

            chain_group['theo_ivs'] = self.get_model_values(chain_group['log_moneyness'], calibration_group)
            chain_group['theos'] = vectorized_black(chain_group['option_types'], chain_group['forward_prices'],
                                                    chain_group['strikes'], chain_group['tau'], chain_group['rates'],
                                                    chain_group['theo_ivs'], return_as='numpy') * np.exp(
                -chain_group['rates'] * chain_group['tau'])
            results.append(chain_group)

        chain_obj.chain = pd.concat(results)
        return chain_obj





class WingModel(object):

    def skew(self, moneyness: ndarray, vc: float, sc: float, pc: float, cc: float, dc: float, uc: float, dsm: float,
             usm: float) -> ndarray:
        """

        :param moneyness: converted strike, moneyness
        :param vc:
        :param sc:
        :param pc:
        :param cc:
        :param dc:
        :param uc:
        :param dsm:
        :param usm:
        :return:
        """
        assert -1 < dc < 0
        assert dsm > 0
        assert 1 > uc > 0
        assert usm > 0
        assert 1e-6 < vc < 10  # 数值优化过程稳定
        assert -1e6 < sc < 1e6
        assert dc * (1 + dsm) <= dc <= 0 <= uc <= uc * (1 + usm)

        # volatility at this converted strike, vol(x) is then calculated as follows:
        vol_list = []
        for x in moneyness:
            # volatility at this converted strike, vol(x) is then calculated as follows:
            if x < dc * (1 + dsm):
                vol = vc + dc * (2 + dsm) * (sc / 2) + (1 + dsm) * pc * pow(dc, 2)
            elif dc * (1 + dsm) < x <= dc:
                vol = vc - (1 + 1 / dsm) * pc * pow(dc, 2) - sc * dc / (2 * dsm) + (1 + 1 / dsm) * (
                        2 * pc * dc + sc) * x - (pc / dsm + sc / (2 * dc * dsm)) * pow(x, 2)
            elif dc < x <= 0:
                vol = vc + sc * x + pc * pow(x, 2)
            elif 0 < x <= uc:
                vol = vc + sc * x + cc * pow(x, 2)
            elif uc < x <= uc * (1 + usm):
                vol = vc - (1 + 1 / usm) * cc * pow(uc, 2) - sc * uc / (2 * usm) + (1 + 1 / usm) * (
                        2 * cc * uc + sc) * x - (cc / usm + sc / (2 * uc * usm)) * pow(x, 2)
            elif uc * (1 + usm) < x:
                vol = vc + uc * (2 + usm) * (sc / 2) + (1 + usm) * cc * pow(uc, 2)
            else:
                raise ValueError("x value error!")
            vol_list.append(vol)
        return array(vol_list)

    def loss_skew(self, params: [float, float, float], x: ndarray, iv: ndarray, vega: ndarray, vc: float, dc: float,
                  uc: float, dsm: float, usm: float, bid=None, ask=None):
        """

        :param params: sc, pc, cc
        :param x:
        :param iv:
        :param vega:
        :param vc:
        :param dc:
        :param uc:
        :param dsm:
        :param usm:
        :return:
        """
        sc, pc, cc = params

        value = self.skew(x, vc, sc, pc, cc, dc, uc, dsm, usm)

        if bid is None:
            vega = vega / vega.max()
            return norm((value - iv) * vega, ord=2, keepdims=False)
        else:
            return norm((value - iv) * (1 / (ask - bid) ** 2), ord=2, keepdims=False)
            # return norm((value - iv)*(1/(np.exp((ask-bid)))**2),ord=2,keepdims=False)

    def loss_skew_with_vc(self, params: [float, float, float, float], x: ndarray, iv: ndarray, vega: ndarray, dc: float,
                          uc: float, dsm: float, usm: float, bid=None, ask=None):

        sc, pc, cc, vc = params
        value = self.skew(x, vc, sc, pc, cc, dc, uc, dsm, usm)
        if bid is None:
            return norm((value - iv), ord=2, keepdims=False)
        else:
            return norm((value - iv) * (1 / (ask - bid) ** 2), ord=2, keepdims=False)

    def calibrate_skew(self, x: ndarray, iv: ndarray, vega: ndarray, dc: float = -0.2, uc: float = 0.2,
                       dsm: float = 0.5,
                       usm: float = 0.5, is_bound_limit: bool = False,
                       epsilon: float = 1e-16, inter: str = "cubic"):
        """

        :param x: moneyness
        :param iv:
        :param vega:
        :param dc:
        :param uc:
        :param dsm:
        :param usm:
        :param is_bound_limit:
        :param epsilon:
        :param inter: cubic inter
        :return:
        """

        vc = interp1d(x, iv, kind=inter, fill_value="extrapolate")([0])[0]

        # init guess for sc, pc, cc
        if is_bound_limit:
            bounds = [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3)]
        else:
            bounds = [(None, None), (None, None), (None, None)]
        initial_guess = normal(size=3)

        args = (x, iv, vega, vc, dc, uc, dsm, usm)
        residual = minimize(self.loss_skew, initial_guess, args=args, bounds=bounds, tol=epsilon, method="SLSQP")
        assert residual.success
        return residual.x, residual.fun

    def sc(self, sr: float, scr: float, ssr: float, ref: float, atm: ndarray or float) -> ndarray or float:
        return sr - scr * ssr * ((atm - ref) / ref)

    def loss_scr(self, x: float, sr: float, ssr: float, ref: float, atm: ndarray, sc: ndarray) -> float:
        return norm(sc - self.sc(sr, x, ssr, ref, atm), ord=2, keepdims=False)

    def fit_scr(self, sr: float, ssr: float, ref: float, atm: ndarray, sc: ndarray, epsilon: float = 1e-16) -> [float,
                                                                                                                float]:
        init_value = array([0.01])
        residual = minimize(self.loss_scr, init_value, args=(sr, ssr, ref, atm, sc), tol=epsilon, method="SLSQP")
        assert residual.success
        return residual.x, residual.fun

    def vc(self, vr: float, vcr: float, ssr: float, ref: float, atm: ndarray or float) -> ndarray or float:
        return vr - vcr * ssr * ((atm - ref) / ref)

    def loss_vc(self, x: float, vr: float, ssr: float, ref: float, atm: ndarray, vc: ndarray) -> float:
        return norm(vc - self.vc(vr, x, ssr, ref, atm), ord=2, keepdims=False)

    def fit_vcr(self, vr: float, ssr: float, ref: float, atm: ndarray, vc: ndarray, epsilon: float = 1e-16) -> [float,
                                                                                                                float]:
        init_value = array([0.01])
        residual = minimize(self.loss_vc, init_value, args=(vr, ssr, ref, atm, vc), tol=epsilon, method="SLSQP")
        assert residual.success
        return residual.x, residual.fun

    def wing(self, x: ndarray, ref: float, atm: float, vr: float, vcr: float, sr: float, scr: float, ssr: float,
             pc: float, cc: float, dc: float, uc: float, dsm: float, usm: float) -> ndarray:
        """
        wing model

        :param x:
        :param ref:
        :param atm:
        :param vr:
        :param vcr:
        :param sr:
        :param scr:
        :param ssr:
        :param pc:
        :param cc:
        :param dc:
        :param uc:
        :param dsm:
        :param usm:
        :return:
        """
        vc = self.vc(vr, vcr, ssr, ref, atm)
        sc = self.sc(sr, scr, ssr, ref, atm)
        return self.skew(x, vc, sc, pc, cc, dc, uc, dsm, usm)


class ArbitrageFreeWingModel(WingModel):

    def calibrate(self, x: ndarray, iv: ndarray, vega: ndarray, dc: float = -0.2, uc: float = 0.2, dsm: float = 0.5,
                  usm: float = 0.5, is_bound_limit: bool = False, epsilon: float = 1e-16, inter: str = "cubic",
                  level: float = 0, method: str = "SLSQP", epochs: int = None, show_error: bool = False,
                  use_constraints: bool = False, bid=None, ask=None, bounds=None, initial_guess=None) -> (
    [float, float, float], float):
        """

        :param x:
        :param iv:
        :param vega:
        :param dc:
        :param uc:
        :param dsm:
        :param usm:
        :param is_bound_limit:
        :param epsilon:
        :param inter:
        :param level:
        :param method:
        :param epochs:
        :param show_error:
        :param use_constraints:
        :return:
        """
        vega = clip(vega, 1e-6, 1e6)
        iv = clip(iv, 1e-6, 10)

        # init guess for sc, pc, cc
        if is_bound_limit:
            if bounds:
                bounds = bounds
            else:
                bounds = [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3)]

        else:
            bounds = [(None, None), (None, None), (None, None)]

        vc = interp1d(x, iv, kind=inter, fill_value="extrapolate")([0])[0]
        constraints = dict(type='ineq', fun=partial(self.constraints, args=(x, vc, dc, uc, dsm, usm), level=level))
        args = (x, iv, vega, vc, dc, uc, dsm, usm)

        if initial_guess:
            guess = initial_guess
        else:
            guess = normal(size=3)

        if epochs is None:
            if use_constraints:

                if bid is None:

                    residual = minimize(self.loss_skew, guess, args=args, bounds=bounds,
                                        constraints=constraints,
                                        tol=epsilon, method=method)

                else:
                    args = (x, iv, vega, vc, dc, uc, dsm, usm, bid, ask)
                    residual = minimize(self.loss_skew, guess, args=args, bounds=bounds,
                                        constraints=constraints,
                                        tol=epsilon, method=method)


            else:

                if bid is None:
                    residual = minimize(self.loss_skew, guess, args=args, bounds=bounds, tol=epsilon,
                                        method=method)
                else:
                    args = (x, iv, vega, vc, dc, uc, dsm, usm, bid, ask)
                    residual = minimize(self.loss_skew, guess, args=args, bounds=bounds, tol=epsilon,
                                        method=method)

            if residual.success:
                sc, pc, cc = residual.x
                arbitrage_free = self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc)
                return residual.x, vc, residual.fun, arbitrage_free
            else:
                epochs = 10
                if show_error:
                    print("calibrate wing-model wrong, use epochs = 10 to find params! params: {}".format(residual.x))

        if epochs is not None:
            params = zeros([epochs, 3])
            loss = ones([epochs, 1])
            for i in range(epochs):
                if use_constraints:
                    residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds,
                                        constraints=constraints,
                                        tol=epsilon, method="SLSQP")
                else:
                    residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds, tol=epsilon,
                                        method="SLSQP")
                if not residual.success and show_error:
                    print("calibrate wing-model wrong, wrong @ {} /10! params: {}".format(i, residual.x))
                params[i] = residual.x
                loss[i] = residual.fun
            min_idx = argmin(loss)
            sc, pc, cc = params[min_idx]
            loss = loss[min_idx][0]
            arbitrage_free = self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc)
            return (sc, pc, cc), vc, loss, arbitrage_free

    def constraints(self, x: [float, float, float], args: [ndarray, float, float, float, float, float],
                    level: float = 0) -> float:
        """蝶式价差无套利约束

        :param x: guess values, sc, pc, cc
        :param args:
        :param level:
        :return:
        """
        sc, pc, cc = x
        moneyness, vc, dc, uc, dsm, usm = args

        if level == 0:
            pass
        elif level == 1:
            moneyness = arange(-1, 1.01, 0.01)
        else:
            moneyness = arange(-1, 1.001, 0.001)

        return self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, moneyness, vc)

    """蝶式价差无套利约束条件
    """

    def left_parabolic(self, sc: float, pc: float, x: float, vc: float) -> float:
        """

        :param sc:
        :param pc:
        :param x:
        :param vc:
        :return:
        """
        return pc - 0.25 * (sc + 2 * pc * x) ** 2 * (0.25 + 1 / (vc + sc * x + pc * x * x)) + (
                1 - 0.5 * x * (sc + 2 * pc * x) / (vc + sc * x + pc * x * x)) ** 2

    def right_parabolic(self, sc: float, cc: float, x: float, vc: float) -> float:
        """

        :param sc:
        :param cc:
        :param x:
        :param vc:
        :return:
        """
        return cc - 0.25 * (sc + 2 * cc * x) ** 2 * (0.25 + 1 / (vc + sc * x + cc * x * x)) + (
                1 - 0.5 * x * (sc + 2 * cc * x) / (vc + sc * x + cc * x * x)) ** 2

    def left_constant_level(self) -> float:
        return 1

    def right_constant_level(self) -> float:
        return 1

    def _check_butterfly_arbitrage(self, sc: float, pc: float, cc: float, dc: float, dsm: float, uc: float, usm: float,
                                   x: float, vc: float) -> float:
        """检查是否存在蝶式价差套利机会，确保拟合time-slice iv-curve 是无套利（无蝶式价差静态套利）曲线

        :param sc:
        :param pc:
        :param cc:
        :param dc:
        :param dsm:
        :param uc:
        :param usm:
        :param x:
        :param vc:
        :return:
        """
        # if x < dc * (1 + dsm):
        #     return self.left_constant_level()
        # elif dc * (1 + dsm) < x <= dc:
        #     return self.left_smoothing_range(sc, pc, dc, dsm, x, vc)
        # elif dc < x <= 0:
        #     return self.left_parabolic(sc, pc, x, vc)
        # elif 0 < x <= uc:
        #     return self.right_parabolic(sc, cc, x, vc)
        # elif uc < x <= uc * (1 + usm):
        #     return self.right_smoothing_range(sc, cc, uc, usm, x, vc)
        # elif uc * (1 + usm) < x:
        #     return self.right_constant_level()
        # else:
        #     raise ValueError("x value error!")

        if dc < x <= 0:
            return self.left_parabolic(sc, pc, x, vc)
        elif 0 < x <= uc:
            return self.right_parabolic(sc, cc, x, vc)
        else:
            return 0

    def check_butterfly_arbitrage(self, sc: float, pc: float, cc: float, dc: float, dsm: float, uc: float, usm: float,
                                  moneyness: ndarray, vc: float) -> float:
        """

        :param sc:
        :param pc:
        :param cc:
        :param dc:
        :param dsm:
        :param uc:
        :param usm:
        :param moneyness:
        :param vc:
        :return:
        """
        con_arr = []
        for x in moneyness:
            con_arr.append(self._check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc))
        con_arr = array(con_arr)
        if (con_arr >= 0).all():
            return minimum(con_arr.mean(), 1e-7)
        else:
            return maximum((con_arr[con_arr < 0]).mean(), -1e-7)

    def left_smoothing_range(self, sc: float, pc: float, dc: float, dsm: float, x: float, vc: float) -> float:
        a = - pc / dsm - 0.5 * sc / (dc * dsm)

        b1 = -0.25 * ((1 + 1 / dsm) * (2 * dc * pc + sc) - 2 * (pc / dsm + 0.5 * sc / (dc * dsm)) * x) ** 2
        b2 = -dc ** 2 * (1 + 1 / dsm) * pc - 0.5 * dc * sc / dsm + vc + (1 + 1 / dsm) * (2 * dc * pc + sc) * x - (
                pc / dsm + 0.5 * sc / (dc * dsm)) * x ** 2
        b2 = (0.25 + 1 / b2)
        b = b1 * b2

        c1 = x * ((1 + 1 / dsm) * (2 * dc * pc + sc) - 2 * (pc / dsm + 0.5 * sc / (dc * dsm)) * x)
        c2 = 2 * (-dc ** 2 * (1 + 1 / dsm) * pc - 0.5 * dc * sc / dsm + vc + (1 + 1 / dsm) * (2 * dc * pc + sc) * x - (
                pc / dsm + 0.5 * sc / (dc * dsm)) * x ** 2)
        c = (1 - c1 / c2) ** 2
        return a + b + c

    def right_smoothing_range(self, sc: float, cc: float, uc: float, usm: float, x: float, vc: float) -> float:
        a = - cc / usm - 0.5 * sc / (uc * usm)

        b1 = -0.25 * ((1 + 1 / usm) * (2 * uc * cc + sc) - 2 * (cc / usm + 0.5 * sc / (uc * usm)) * x) ** 2
        b2 = -uc ** 2 * (1 + 1 / usm) * cc - 0.5 * uc * sc / usm + vc + (1 + 1 / usm) * (2 * uc * cc + sc) * x - (
                cc / usm + 0.5 * sc / (uc * usm)) * x ** 2
        b2 = (0.25 + 1 / b2)
        b = b1 * b2

        c1 = x * ((1 + 1 / usm) * (2 * uc * cc + sc) - 2 * (cc / usm + 0.5 * sc / (uc * usm)) * x)
        c2 = 2 * (-uc ** 2 * (1 + 1 / usm) * cc - 0.5 * uc * sc / usm + vc + (1 + 1 / usm) * (2 * uc * cc + sc) * x - (
                cc / usm + 0.5 * sc / (uc * usm)) * x ** 2)
        c = (1 - c1 / c2) ** 2
        return a + b + c


class OrcWingModelCalibrate:

    def __init__(self):
        pass

    def calibrate_model_params_to_chain(self, chain_obj, y_col='mid_iv', initial_guess=None,
                                        recalibrate_with_respect_to=['sc', 'vc', 'cc', 'pc'],
                                        recalibrate_arb_free=False,
                                        obj_func='gatheral', arbitrage_free=True, level=0, params_buffer=None,url=None):
        if url:
            item = self.calibrate_model_params_to_chain_from_url(chain_obj=chain_obj,url=url,y_col=y_col,recalibrate_with_respect_to=recalibrate_with_respect_to,recalibrate_arb_free=recalibrate_arb_free,obj_func=obj_func,arbitrage_free=arbitrage_free,level=level,params_buffer=params_buffer,initial_guess=initial_guess)
            return item


        from copy import deepcopy
        chain_obj = deepcopy(chain_obj)
        chain_obj.calculate_needed_col(['log_moneyness', y_col])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)

        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])

        arbitrage_free_wing_model = ArbitrageFreeWingModel()

        if initial_guess is None:
            params_dict = {}
            dc, uc, dsm, usm = -.999, .999, 400, 400
            #dc, uc, dsm, usm = -.25, .25, .5, .5
        else:
            item = self.recalibrate_model(chain_obj, initial_guess)  # , calibrate_with_respect_to, change_scr_vcr,
            #jl.GC.enable()
            return item
            # obj_func, arbitrage_free, level, params_buffer)

        for (ticker, expiration_date), group in grouped:
            if 'filter' in group.columns:
                group = group[group['filter'] == False]

            if arbitrage_free:
                use_constraints = True
            else:
                use_constraints = False
            if obj_func == 'gatheral':
                bid = group['bid_iv']
                ask = group['ask_iv']
            else:
                bid = None
                ask = None

            r = time.time()
            params, vc, loss, arb_free = arbitrage_free_wing_model.calibrate(group['log_moneyness'], group[y_col],
                                                                             pd.Series(), dc, uc,
                                                                             dsm, usm, is_bound_limit=True,
                                                                             use_constraints=use_constraints,
                                                                             level=level,
                                                                             bid=bid, ask=ask)

            sc, pc, cc = params
            if ticker not in params_dict.keys():
                params_dict[ticker] = {}

            params_dict[ticker][expiration_date] = {}
            params_dict[ticker][expiration_date]['dc'] = dc
            params_dict[ticker][expiration_date]['uc'] = uc
            params_dict[ticker][expiration_date]['dsm'] = dsm
            params_dict[ticker][expiration_date]['usm'] = usm
            params_dict[ticker][expiration_date]['vr'] = vc
            params_dict[ticker][expiration_date]['vcr'] = 0
            params_dict[ticker][expiration_date]['vc'] = vc
            params_dict[ticker][expiration_date]['sr'] = sc
            params_dict[ticker][expiration_date]['scr'] = 0
            params_dict[ticker][expiration_date]['sc'] = sc
            params_dict[ticker][expiration_date]['ssr'] = 100
            params_dict[ticker][expiration_date]['y_col'] = y_col
            params_dict[ticker][expiration_date]['obj_func'] = obj_func
            params_dict[ticker][expiration_date]['pc'] = pc
            params_dict[ticker][expiration_date]['cc'] = cc

            params_dict[ticker][expiration_date]['ATM'] = group['forward_prices'].iloc[0]
            params_dict[ticker][expiration_date]['REF'] = group['forward_prices'].iloc[0]
            params_dict[ticker][expiration_date]['recalibrate_with_respect_to'] = recalibrate_with_respect_to
            params_dict[ticker][expiration_date]['recalibrate_arb_free'] = recalibrate_arb_free

        return params_dict

    def recalibrate_model(self, chain_obj, params_):  # edit to include ensure arbitrage by modifying param bounds
        # possibly reuse old vc value if obj func loss is greater? or maybe recalibrate it if that works? idek. spreads widening will be annoying
        from copy import deepcopy
        from py_vollib_vectorized import vectorized_black
        chain_obj = deepcopy(chain_obj)
        chain_obj.calculate_needed_col(['log_moneyness'])
        chain_obj.chain = chain_obj.chain.sort_values(['ticker', 'expiration_dates', 'strikes']).reset_index(drop=True)

        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        arbitrage_free_wing_model = ArbitrageFreeWingModel()

        params_dict = {}

        for (ticker, expiration_date), group in grouped:
            group = group.sort_values('log_moneyness').reset_index(drop=True)
            params_for_chain = deepcopy(params_[ticker][expiration_date])
            recalibration_params = params_for_chain['recalibrate_with_respect_to']
            pc, cc = params_for_chain['pc'], params_for_chain['cc']
            bounds = self.get_bounds(pc, cc, recalibration_params)
            if 'filter' in group.columns:
                #print(f'b4: {len(group)}')

                group = group[group['filter'] == False]
                #print(f'after: {len(group)}')
            x = group['log_moneyness']
            y_col = params_for_chain['y_col']
            iv = group[y_col]
            if params_for_chain['obj_func'] == 'gatheral':
                vega = pd.Series()
                bid = group['bid_iv']
                ask = group['ask_iv']
            else:
                vega = group['vega']
                bid = None
                ask = None

            dc, uc, dsm, usm, sc_old, pc_old, cc_old, arb_free = params_for_chain['dc'], params_for_chain['uc'], \
            params_for_chain['dsm'], \
                params_for_chain['usm'], params_for_chain['sc'], params_for_chain['pc'], params_for_chain['cc'], \
            params_for_chain['recalibrate_arb_free']

            #s = time.time()
            params, vc, loss, arbitrage_free = arbitrage_free_wing_model.calibrate(x=x, iv=iv, vega=vega, dc=dc, uc=uc,
                                                                             dsm=dsm,
                                                                             usm=usm, is_bound_limit=True,
                                                                             bounds=bounds,
                                                                             use_constraints=arb_free, bid=bid, ask=ask,
                                                                             initial_guess=[sc_old, pc_old, cc_old])


            sc, pc, cc = params

            #print(sc,pc,cc,dc,dsm,uc,usm,vc,arbitrage_free)
            #print(group['log_moneyness'])


            sr, ssr, ref, vr = params_for_chain['sr'], params_for_chain['ssr'], params_for_chain['REF'], \
                params_for_chain['vr']
            new_atm = group['forward_prices'].iloc[0]

            # if new_atm == ref:
            #    sc = params_for_chain['sc']
            #    scr = params_for_chain['scr']
            # else:
            scr, _ = arbitrage_free_wing_model.fit_scr(sr, ssr, ref, new_atm, sc)
            scr = scr[0]

            vcr, _ = arbitrage_free_wing_model.fit_vcr(vr, ssr, ref, new_atm, vc)
            vcr = vcr[0]

            #sc_ = arbitrage_free_wing_model.sc(sr=sr, scr=scr, ssr=ssr, ref=ref, atm=new_atm)
            sc_ = sc
            #vc_ = arbitrage_free_wing_model.vc(vr, vcr, ssr, ref, new_atm)
            vc_=vc

            if ticker not in params_dict.keys():
                params_dict[ticker] = {}

            params_dict[ticker][expiration_date] = {}
            params_dict[ticker][expiration_date]['dc'] = dc
            params_dict[ticker][expiration_date]['uc'] = uc
            params_dict[ticker][expiration_date]['dsm'] = dsm
            params_dict[ticker][expiration_date]['usm'] = usm
            params_dict[ticker][expiration_date]['vr'] = vr
            params_dict[ticker][expiration_date]['vcr'] = vcr
            params_dict[ticker][expiration_date]['vc'] = vc_
            params_dict[ticker][expiration_date]['sr'] = sr
            params_dict[ticker][expiration_date]['scr'] = scr
            params_dict[ticker][expiration_date]['sc'] = sc_
            params_dict[ticker][expiration_date]['ssr'] = 100
            params_dict[ticker][expiration_date]['y_col'] = y_col
            params_dict[ticker][expiration_date]['obj_func'] = params_for_chain['obj_func']
            params_dict[ticker][expiration_date]['pc'] = pc
            params_dict[ticker][expiration_date]['cc'] = cc

            params_dict[ticker][expiration_date]['ATM'] = new_atm
            params_dict[ticker][expiration_date]['REF'] = ref
            params_dict[ticker][expiration_date]['recalibrate_with_respect_to'] = recalibration_params
            params_dict[ticker][expiration_date]['recalibrate_arb_free'] = arb_free


        return params_dict

    def evaluate_model_for_chain(self, chain_obj, params):
        from py_vollib_vectorized import vectorized_black
        chain_obj = deepcopy(chain_obj)
        grouped = chain_obj.chain.groupby(['ticker', 'expiration_dates'])
        arbitrage_free_wing_model = ArbitrageFreeWingModel()
        new_chains = []

        for (ticker, expiration_date), group in grouped:
            params_for_exp = params[ticker][expiration_date]
            x, ref, atm, vr, vcr, sr, scr, ssr, pc, cc, dc, uc, dsm, usm = group['log_moneyness'], params_for_exp[
                'REF'], group['forward_prices'].iloc[0], params_for_exp['vr'], params_for_exp['vcr'], params_for_exp[
                'sr'], params_for_exp['scr'], params_for_exp['ssr'], params_for_exp['pc'], params_for_exp['cc'], \
                params_for_exp['dc'], params_for_exp['uc'], params_for_exp['dsm'], params_for_exp['usm']
          #  group['theo_ivs'] = arbitrage_free_wing_model.wing(x=x, ref=ref, atm=atm, vr=vr, vcr=vcr, sr=sr, scr=scr,
           #                                                    ssr=ssr, pc=pc, cc=cc, dc=dc, uc=uc, dsm=dsm, usm=usm)

            #print(vr, vcr, sr, scr)


            vc,sc = params_for_exp['vc'],params_for_exp['sc']
            #print(vc,sc)

            group['theo_ivs'] = arbitrage_free_wing_model.skew(moneyness=x,vc=vc,sc=sc,pc=pc,cc=cc,dc=dc,uc=uc,dsm=dsm,usm=usm)
            new_chains.append(group)

        new_chain = pd.concat(new_chains)

        new_chain['theos'] = vectorized_black(new_chain['option_types'], new_chain['forward_prices'],
                                              new_chain['strikes'], new_chain['tau'], new_chain['rates'],
                                              new_chain['theo_ivs'], return_as='numpy') * np.exp(
            -new_chain['rates'] * new_chain['tau'])
        chain_obj.chain = new_chain
        return chain_obj

    def calibrate_scr_in_params(self, chain, params, new_scr):
        pass

    def calibrace_vcr_in_params(self, chain, params, new_scr):
        pass

    def get_bounds(self, pc, cc, recalibrate_with_respect_to, params_buffer=None):

        params_not_to_calibrate = list(set(['cc', 'pc']).difference(set(recalibrate_with_respect_to)))
        if len(params_not_to_calibrate) == 0:
            return [(-1e5, 1e5), (-1e7, 1e7), (-1e7, 1e7)]
        else:
            if 'cc' in params_not_to_calibrate:
                cc_bounds = (cc, cc)
            else:
                cc_bounds = (-1e7, 1e7)
            if 'pc' in params_not_to_calibrate:
                pc_bounds = (pc, pc)
            else:
                pc_bounds = (-1e7, 1e7)

            return [(-1e5, 1e5), pc_bounds, cc_bounds]
    def make_calibration_result_jsonable(self,result):
        #none needed for this
        return result

    def revert_calibration_input(self,data):
        return data

    def calibrate_model_params_to_chain_from_url(self, chain_obj,url,y_col, initial_guess,
                                        recalibrate_with_respect_to,
                                        recalibrate_arb_free,
                                        obj_func, arbitrage_free, level, params_buffer):

        import requests
        df = make_json_serializable(chain_obj.chain)
        # url = "http://0.0.0.0:8000/calibration"
        url = f"{url}/calibration"
        payload = {
            "data": {
                'chain': df.to_dict(),
                'model_obj_str': 'OWM',
                'params': {
                    'y_col': y_col,
                    'initial_guess': initial_guess,
                    'obj_func': obj_func,
                    'recalibrate_with_respect_to':recalibrate_with_respect_to,
                    'recalibrate_arb_free':recalibrate_arb_free,
                    'arbitrage_free':arbitrage_free,
                    'level':level,
                    'params_buffer':params_buffer

                }
            }
        }

        response = requests.post(url, json=payload).json()
        response = {ticker: {pd.to_datetime(expo): val for expo, val in exp_dict.items()} for ticker, exp_dict in
                    response.items()}

        return response



class StochasticCollocationBSplineCalibrate:
    def __init__(self):
        pass


'''
df = pd.read_csv('/Users/lakaj/Documents/df_tst3.csv')
df = df[(df['snap_shot_datetimes'] == df['snap_shot_datetimes'].iloc[0]) & (df['expiration_dates'] == df['expiration_dates'].iloc[0])]
svi = SSVI()
chain_obj = ChainReinit(df)
init = svi.calibrate_model_params_to_chain(chain_obj,obj_func='gatheral')
print(init)
print(svi.recalibrate_model(chain_obj,init))

'''





'''
model = VolatilityModel()
n = 6 # Number of parameters to calibrate, with 3 + 2*(n-3) parameters in total

s = time.time()
# Fit the model to the chain and add the model IV column
model.fit_and_add_model_iv(chain, n)


# Plot the results
expirations = chain.chain['expiration_dates'].unique()
for expiration in expirations:
    subset = chain.chain[chain.chain['expiration_dates'] == expiration]
    plt.figure()
    plt.scatter(subset['strikes'], subset['bid_iv'], color='blue', label='Bid IV', s=2)
    plt.scatter(subset['strikes'], subset['ask_iv'], color='red', label='Ask IV', s=2)
    plt.plot(subset['strikes'], subset['model_iv'], 'g-', label='Model IV')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Expiration: {expiration}')
    plt.legend()
    plt.show()






class CalibrationResult:
    def __init__(self, params, additional_info=None):
        self.params = params  # This could be a dictionary, DataFrame, etc.
        self.additional_info = additional_info  # Any other data needed





class BaseModel(ABC): # ABC makes it so that this class can't be initialized
    def __init__(self):
        self.config = {}
        self.latest_calibration = None

    @abstractmethod #declares below function must belong to all subclasses
    def calibrate_model_params_to_chain(self, chain_obj, **kwargs):
        self.config['calibration'] = kwargs #stores calibration keyword arguments

    @abstractmethod
    def evaluate_model_for_chain(self, chain_obj, **kwargs):
        self.config['evaluation'] = kwargs

    @abstractmethod
    def price_chain(self, chain_obj, **kwargs):
        self.config['pricing'] = kwargs

class HestonModel(BaseModel):
    def __init__(self, params):
        super().__init__() #initializes the parent class
        self.params = params

    def calibrate_model_params_to_chain(self, chain_obj, **kwargs):
        super().calibrate_model_params_to_chain(chain_obj, **kwargs)
        # Heston-specific calibration logic here
        
        # Assume calibration returns a DataFrame or similar structure
        calibration_results = CalibrationResult(params={'kappa': 0.1}, additional_info={'iteration': 1})
        self.latest_calibration = calibration_results
        return calibration_results

    def evaluate_model_for_chain(self, chain_obj, **kwargs):
        super().evaluate_model_for_chain(chain_obj, **kwargs)
        PEERINT(f"Evaluating with settings: {self.config['evaluation']}")

    def price_chain(self, chain_obj, **kwargs):
        super().price_chain(chain_obj, **kwargs)
        PEERINT(f"Pricing with settings: {self.config['pricing']}")

class ModelHandler:
    def __init__(self, model):
        self.model = model

    def calibrate(self, chain_obj, use_previous_results=True):
        calibration_kwargs = self.model.config.get('calibration', {})
        if use_previous_results and self.model.latest_calibration:
            calibration_kwargs.update({'initial_guess': self.model.latest_calibration.params})
        return self.model.calibrate_model_params_to_chain(chain_obj, **calibration_kwargs)

    def evaluate(self, chain_obj):
        return self.model.evaluate_model_for_chain(chain_obj, **self.model.config.get('evaluation', {}))

    def price(self, chain_obj):
        return self.model.price_chain(chain_obj, **self.model.config.get('pricing', {}))

'''












