import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from ray.util.joblib import register_ray
from dask.distributed import Client
import warnings


class PairsTrading:
    def __init__(self, position_value, freq='daily',
                 hr_test='1', hr_trade = 'KF', hr_constant=False, hr_window=30, hr_min_share_decimal=4,
                 perf_signal='zscore', perf_signal_window=30, perf_signal_entry=2, perf_signal_exit=0,
                 perf_daily_rebalance=False, perf_theoretical_adf=False,
                 n_jobs=-1, backend='loky', hide_progress_bar = False):
        """
        backend: loky, ray, dask, multiprocessing, threading
        """
        if backend == 'dask':
            Client()
        elif backend == 'ray':
            register_ray()
        self.position_value = position_value
        self.freq = freq
        self.hr_test = hr_test
        self.hr_trade = hr_trade
        self.hr_constant = hr_constant
        self.hr_window = hr_window
        self.hr_min_share_decimal = hr_min_share_decimal
        self.perf_signal = perf_signal
        self.perf_signal_window = perf_signal_window
        self.perf_signal_entry = perf_signal_entry
        self.perf_signal_exit = perf_signal_exit
        self.perf_daily_rebalance = perf_daily_rebalance
        self.perf_theoretical_adf = perf_theoretical_adf
        self.n_jobs = n_jobs
        self.backend = backend
        self.no_tqdm_bar = hide_progress_bar

    @staticmethod
    def get_cointegrated_pairs(data, p_cutoff=0.05):
        """
        Not to be used due to poor performance.
        Test pairwise co-integration using Engle-Granger two step procedure. This method has lots of false positives if
        hedge ratio is not generated by OLS.
        """

        def get_Engle_Granger_pval(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None):
            return sm.tsa.stattools.coint(y0, y1, trend, method, maxlag, autolag, return_results)[1]

        pvalue_df = data.corr(method=get_Engle_Granger_pval)
        upper_matrix = np.triu(np.ones(pvalue_df.shape), k=1).astype(np.bool)
        pvalue_df = pvalue_df.where(upper_matrix).unstack().sort_values(ascending=False)
        pvalue_df = pvalue_df[pvalue_df <= p_cutoff].to_frame(name='p_val')
        return pvalue_df

    @staticmethod
    def get_pairs(data, pairs=None):
        """
        get initial pairs from the data
        """
        if pairs is None:
            ncol = len(data.columns)
            cols = []
            for i in range(ncol):
                for j in range(i + 1, ncol):
                    cols.append((data.columns[j], data.columns[i]))
            pairs = cols
        elif pairs is not None:
            pairs = pairs
        return pairs

    @staticmethod
    def _KF(X, y, state_mean=0, state_cov=1, trans_var=0.001, observ_var=2, with_constant=False):
        """
        # kalman filter to get hedge ratio
        """
        def KF_with_constant(X, y, state_mean=0, state_cov=0.001, trans_var=0.001, observ_var=2):
            kf = KalmanFilter(
                n_dim_obs=1,
                n_dim_state=2,
                initial_state_mean=[state_mean, state_mean],
                initial_state_covariance=state_cov * np.ones((2, 2)),
                transition_matrices=np.eye(2),
                transition_covariance=trans_var * np.eye(2),
                observation_matrices=np.expand_dims(sm.tools.tools.add_constant(X), axis=1),
                observation_covariance=observ_var)

            state_means, _ = kf.filter(y.values)
            return state_means[:, 1]

        def KF2_without_constant(X, y, state_mean=0, state_cov=1, trans_var=0.001, observ_var=2):
            kf = KalmanFilter(
                n_dim_obs=1,
                n_dim_state=1,
                initial_state_mean=state_mean,
                initial_state_covariance=state_cov,
                transition_matrices=1,
                transition_covariance=trans_var,
                observation_matrices=np.expand_dims(np.expand_dims(X, axis=1), axis=1),
                observation_covariance=[observ_var])

            state_means, _ = kf.filter(y.values)
            return state_means[:, 0]

        if with_constant is True:
            return KF_with_constant(X, y, state_mean, state_cov, trans_var, observ_var)
        elif with_constant is False:
            return KF2_without_constant(X, y, state_mean, state_cov, trans_var, observ_var)

    @staticmethod
    def _OLS(X: pd.Series, y: pd.Series, window: int, with_constant:bool) -> float:
        """
        # rolling OLS to get hedge ratio
        """
        if window <= len(y):
            X = sm.tools.tools.add_constant(X) if with_constant is True else X
            return RollingOLS(y, X, window).fit().params.values[:, 0]
        else:
            warnings.warn('windows size larger than the length of data. return np.nan')
            return np.nan


    @staticmethod
    def _get_half_life(spread):
        """
        half life for spread, which is used as the look-back horizon to calculate z_score
        """
        spread_lag = spread.shift(1).dropna()
        spread_delta = spread.diff(1).dropna()
        model = sm.OLS(spread_delta, sm.add_constant(spread_lag))
        res = model.fit()
        halflife = int(-np.log(2) / res.params[1])
        if halflife <= 0:
            halflife = np.nan
        return halflife


    def _backtest_pair(self, X, y,**kl_kwargs):
        """
        back test a single pair to get hedge ratio, spread, adf
        hr: hedge ratio
        """
        # construct df
        col_x = X.name
        col_y = y.name
        df = pd.DataFrame()
        df[col_x] = X
        df[col_y] = y
        # regression to get hedge ratio, spread, and pval for a single pair
        if self.hr_test == 'KF':
            hr_test = self._KF(X, y, with_constant=self.hr_constant, **kl_kwargs)
        elif self.hr_test == 'OLS':
            hr_test = self._OLS(X, y, window=self.hr_window, with_constant=self.hr_constant)
        elif self.hr_test == '1':
            hr_test = 1
        else:
            raise ValueError('Incorrect hr_method value. Valid values are KF or OLS')
        # return hedge ratio and spread
        df['hr_test'] = hr_test
        df['spread_test'] = y - hr_test * X
        # regression to get hedge ratio, spread, and pval for a single pair
        if self.hr_trade != self.hr_test:
            if self.hr_trade == 'KF':
                hr_trade0 = self._KF(X, y, with_constant=self.hr_constant, **kl_kwargs)
            elif self.hr_trade == 'OLS':
                hr_trade0 = self._OLS(X, y, window=self.hr_window, with_constant=self.hr_constant)
            elif self.hr_trade == '1':
                hr_trade0 = 1
            else:
                raise ValueError('Incorrect hr_method value. Valid values are KF or OLS')
        else:
            hr_trade0 = hr_test
        # effective hedge ratio considering decimal restriction on fractional shares imposed by brokers
        y_n = (self.position_value / y).round(self.hr_min_share_decimal)
        X_n = (y_n * hr_trade0).round(self.hr_min_share_decimal)
        df['hr_trade0'] = X_n / y_n
        # entry and exit rule based on signals
        if self.perf_signal == 'zscore':
            spread_mean = df['spread_test'].rolling(window=self.perf_signal_window).mean()
            spread_std = df['spread_test'].rolling(window=self.perf_signal_window).std()
            df['sig'] = (df['spread_test'] - spread_mean) / spread_std
            long_entry, long_exit = -self.perf_signal_entry, -self.perf_signal_exit
            short_entry, short_exit = self.perf_signal_entry, self.perf_signal_exit
        elif self.perf_signal == 'ecdf':
            df['sig'] = df['spread_test'].rolling(window=self.perf_signal_window).apply(lambda arg: ECDF(arg)(arg[-1]))
            long_entry, long_exit = self.perf_signal_entry, self.perf_signal_exit
            short_entry, short_exit = 1 - self.perf_signal_entry, 1 - self.perf_signal_exit
        else:
            raise ValueError('Invalid signal name. Valid signals are "zscore" or "ecdf"')

        # long entry and exit
        df['long_entry'] = (df['sig'] < long_entry) & (df['sig'].shift(1) > long_entry)
        df['long_exit'] = (df['sig'] > long_exit) & (df['sig'].shift(1) < long_exit)
        df['position_long'] = np.nan
        df.loc[df['long_entry'], 'position_long'] = 1
        df.loc[df['long_exit'], 'position_long'] = 0
        # short entry and exit
        df['short_entry'] = (df['sig'] > short_entry) & (df['sig'].shift(1) < short_entry)
        df['short_exit'] = (df['sig'] < short_exit) & (df['sig'].shift(1) > short_exit)
        df.loc[df['short_entry'], 'position_short'] = -1
        df.loc[df['short_exit'], 'position_short'] = 0
        # additional long exit if short entry happens
        df.loc[df['short_entry'], 'position_long'] = 0
        # additional short exit if long entry happens
        df.loc[df['long_entry'], 'position_short'] = 0
        # fill np.nan
        df['position_long'] = df['position_long'].fillna(method='pad')
        df['position_long'] = df['position_long'].fillna(0)
        df['position_short'] = df['position_short'].fillna(method='pad')
        df['position_short'] = df['position_short'].fillna(0)

        df['position'] = df['position_long'] + df['position_short']


        # actual hedge ratio and spread used in trading.
        if self.perf_daily_rebalance is False:
            df['hr_trade'] = np.nan
            df.loc[df['long_entry'] | df['short_entry'], 'hr_trade'] = df['hr_trade0']
            df['hr_trade'] = df['hr_trade'].fillna(method='pad')
            df['spread_trade'] = df[col_y] - df['hr_trade'] * df[col_x]
        else:
            df['hr_trade'] = df['hr_trade0']
            df['spread_trade'] = df[col_y] - df['hr_trade0'] * df[col_x]
        # calculate return

        pair_r_name = f'r_{col_x}_{col_y}'
        value_today = df[col_y] - df['hr_trade'].shift(1) * df[col_x]
        value_yesterday = df['spread_trade'].shift(1)
        spread_delta = value_today - value_yesterday
        df.loc[df['position'].shift(1) < 0, pair_r_name] = -spread_delta / (df['hr_trade'] * df[col_x]).shift(1)
        df.loc[df['position'].shift(1) > 0, pair_r_name] = spread_delta / df[col_y].shift(1)
        df[pair_r_name] = df[pair_r_name].fillna(0)
        # rename position for later use
        df = df.rename(columns={'position_short': f'position_short_{col_x}_{col_y}',
                                'position_long': f'position_long_{col_x}_{col_y}'})\
                .drop('hr_trade0', axis = 1)
        return df


    def backtest_pairs(self, data, pairs, **kl_kwargs):
        """
        backtest all pairs
        signal: 'zscore' or 'ecdf'
        window: size of the rolling window
        entry:  value of the absolute zscore or the left tail probability if signal is ecdf
        exit: value of exit signal
        """
        def get_perf_single_pair(X, y):
            col_x = X.name
            col_y = y.name
            r = f'r_{col_x}_{col_y}'
            res = self._backtest_pair(X, y, **kl_kwargs)
            days = 252 if self.freq == 'daily' else np.nan
            sr = res[r].mean() / res[r].std() * np.sqrt(days) if res[r].std() != 0 else np.nan
            cum_r = (res[r] + 1).cumprod()[-1]
            p_val0 = sm.tsa.stattools.adfuller(res['spread'].dropna(), regression='c')[
                1] if self.perf_theoretical_adf is True else None
            # exception handling for adf test. Adf test prints error if sample size is too small.
            # sample size could be 0 if there is no trade.
            try:
                p_val = sm.tsa.stattools.adfuller(res['spread_test'].dropna(), regression='c')[1]
            except Exception as e:
                #print(e)
                #print(f'Pair is {col_x} and {col_y}')
                p_val = np.nan
            return {'x': col_x, 'y': col_y,
                    'sharpe_ratio': sr, 'cum_r': cum_r, 'p_val0': p_val0, 'p_val': p_val,
                    'ret': res[r],
                    'position_short': res[f'position_short_{col_x}_{col_y}'],
                    'position_long': res[f'position_long_{col_x}_{col_y}']}

        # get pairs
        pairs = self.get_pairs(data, pairs)
        print(f'Backtesting {len(pairs)} pairs...')
        # get hedge ratio, spread, and performance for all pairs
        with parallel_backend(backend=self.backend):
            results = Parallel(n_jobs=self.n_jobs) \
            (delayed(get_perf_single_pair)(X=data[i], y=data[j])for (i, j) in tqdm(pairs, leave = True, position=0,disable=self.no_tqdm_bar))
        # positions and returns
        r_position = pd.concat([i['ret'] for i in results], axis=1)
        r_position['port_r'] = r_position.sum(axis=1) / len(r_position.columns)
        position_short = pd.concat([i['position_short'] for i in results], axis=1)
        r_position['positions_short'] = position_short.sum(axis=1)
        position_long = pd.concat([i['position_long'] for i in results], axis=1)
        r_position['positions_long'] = position_long.sum(axis=1)
        # performance
        perfs = pd.DataFrame(results)[['x', 'y', 'sharpe_ratio', 'cum_r', 'p_val0', 'p_val']]
        return r_position[['port_r', 'positions_long', 'positions_short']], perfs

    def select_pairs(self, perfs, weight = [0.5, 0.5],  nlargest=20, unique_ticker=True):
        """
        unique_ticker: if set to true, a ticker can only get selected in one pair
        """
        # comnbine signal p_val and sharpe_ratio into zscore to select pairs
        perfs['neg_p_val'] = - perfs['p_val']
        neg_p_val_zscore = (perfs['neg_p_val'] - perfs['neg_p_val'].mean()) / perfs['neg_p_val'].std()
        sr_zscore = (perfs['sharpe_ratio'] - perfs['sharpe_ratio'].mean()) / perfs['sharpe_ratio'].std()
        perfs['zscore'] =  weight[0]*neg_p_val_zscore + weight[1]*sr_zscore

        perfs = perfs.sort_values('zscore', ascending=False)
        # only select pairs with unique tickers
        if unique_ticker is True:
            tickers = perfs[['x', 'y']]
            selected_rows = np.zeros((len(tickers),), dtype=bool)
            current_row = 0
            while selected_rows.sum() < nlargest and current_row < len(perfs):
                selected_tickers = np.unique(tickers.loc[selected_rows].values)
                current_row_tickers = tickers.iloc[current_row, :]
                if current_row_tickers['x'] not in selected_tickers and current_row_tickers['y'] not in selected_tickers:
                    selected_rows[current_row] = True
                current_row = current_row + 1
            perfs = perfs.loc[selected_rows]
        return list(zip(perfs['x'], perfs['y']))

