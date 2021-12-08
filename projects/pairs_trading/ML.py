"""
1.  short training period tends to overfit. And a single stock can repeat appearing in selected pairs. Til now,
    larger stock result in good performance, but smaller stock result in very bad performance.
2. longer period has stable performance and the performance is good even with smaller sample.
3. Imposing selecting unique tickers when selecting by perf generally results in poorer performance compared with
    duplicate tickers.
"""
##### import the necessary modules and set chart style####
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader.data as web
import matplotlib.pylab as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
from math import sqrt
import time
import utilities as utils
import importlib
import pairs_trading

importlib.reload(pairs_trading)
from pairs_trading import PairsTrading
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)
from sklearn.model_selection import ParameterGrid


# remove columns with high percentage of na
def delete_na_cols(data, na_pct):
    for col in data.columns:
        if sum(data[col].isnull()) / len(data) >= na_pct:
            data = data.drop([col], axis=1)
    return data.dropna(axis=0)


import quantstats

# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# read crypto data
crypto = pd.read_pickle('crypto_cmc_daily.pkl').set_index('Date')[['symbol', 'Close']]
crypto = pd.pivot(crypto, columns='symbol', values='Close')
crypto = delete_na_cols(crypto, 0.5)

# read stock data
stock = pd.read_pickle('stock.pkl')
stock_mega = stock.loc[stock['scalemarketcap'] == '6 - Mega']
stock_large = stock.loc[(stock['scalemarketcap'] == '6 - Mega') | (stock['scalemarketcap'] == '5 - Large')]
stock_mid = stock.loc[(stock['scalemarketcap'] == '6 - Mega') |
                      (stock['scalemarketcap'] == '5 - Large') |
                      (stock['scalemarketcap'] == '4 - Mid')]

# train and test
## fixed params
hr = 'KF'
constant = True
metric = 'sharpe_ratio'
signal, entry, exit = 'ecdf', 0.05, 0.5
window = 30
n = 0.5
dataset = stock_mid
# read data
dataset = dataset[['date', 'ticker', 'closeadj']].drop_duplicates(['date', 'ticker'])
data = pd.pivot(dataset, index='date', columns='ticker', values='closeadj').dropna(axis=1)
n_largest = int(0.1 * len(data.columns))
# train test split
train_days, val_days = int(252 * n), 252
train, val = data[-(train_days + val_days * 2):-val_days * 2], data[-val_days * 2:-val_days]
# train
pt = PairsTrading(position_value=10000, hr_method=hr, hr_window=window, hr_min_share_decimal=4,
                  hr_constant=constant, perf_theoretical_adf=True,
                  perf_signal=signal, perf_signal_window=30, perf_signal_entry=entry, perf_signal_exit=exit,
                  perf_daily_rebalance=False, n_jobs=-1, backend='loky')
hrspread = memory.cache(pt.get_spreads)(data=train)
_, perfs = memory.cache(pt.get_perf)(data=train, pairs=None, hr_spread=hrspread)
pairs = list(zip(perfs['x'], perfs['y']))
hrspread2 = pt.get_spreads(val, pairs=pairs)
r_position2, perfs2 = pt.get_perf(data=val, pairs=pairs, hr_spread=hrspread2)
perfs_combined = perfs.merge(perfs2, on=['x', 'y']).drop(['p_val0_x', 'p_val0_y'], axis=1).dropna()
##---OLS
import statsmodels.api as sm

x = perfs_combined[['sharpe_ratio_x', 'cum_r_x', 'p_val_x']]
X = sm.add_constant(X)
Y = perfs_combined['sharpe_ratio_y']
model = sm.OLS(Y, X)
results = model.fit()
results.params
results.tvalues
results.rsquared
## Machine Learning
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error

param_dist = {"max_depth": randint(1, 100),
              "max_features": randint(1, len(x.columns)),
              'min_samples_split': randint(1, 300),
              "min_samples_leaf": randint(1, 300),
              'max_features': uniform(),
              'max_leaf_nodes': randint(1, 300),
              "criterion": ["mse"]}
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
tree_cv.fit(x, Y)
y_pred = tree_cv.predict(x)
test_mse = mean_squared_error(Y, y_pred)
print(f'test mse is {test_mse}')

perfs_combined['pred'] = y_pred
ga = perfs_combined.sort_values('pred').tail(10)
pairs = list(zip(ga['x'], ga['y']))
# get performance on selected pairs
hrspread2 = pt.get_spreads(val, pairs=pairs)
r_position2, _ = pt.get_perf(data=val, pairs=pairs, hr_spread=hrspread2)
r = r_position2['port_r']
quantstats.reports.html(r, 'SPY', output='report2.html')

