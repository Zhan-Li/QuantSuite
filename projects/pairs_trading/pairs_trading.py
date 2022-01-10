"""
1.  short training period tends to overfit. And a single stock can repeat appearing in selected pairs. Til now,
    larger stock result in good performance, but smaller stock result in very bad performance.
2. longer period has stable performance and the performance is good even with smaller sample.
3. Imposing selecting unique tickers when selecting by perf generally results in poorer performance compared with
    duplicate tickers.
4. In hyperparameter tune, never use joblib njob = -1 with ray, or ray idle will stay alive will not release the resources.
"""
import importlib
from datetime import datetime
from math import sqrt

import matplotlib.pylab as plt
##### import the necessary modules and set chart style####
import pandas as pd
import seaborn as sns
from quantsuite.pairs_trading import PairsTrading
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)
from sklearn.model_selection import ParameterGrid


# ray.shutdown()
# ray.init(object_store_memory=10*10**9)
# remove columns with high percentage of na
def delete_na_cols(data, na_pct):
    for col in data.columns:
        if sum(data[col].isnull()) / len(data) >= na_pct:
            data = data.drop([col], axis=1)
    return data


import quantstats
import datetime
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

# ray.init(object_store_memory=10*10**9)
# hyper parameter search algos
ax_search = AxSearch()
bayesopt = BayesOptSearch()
# hebo = HEBOSearch()
hyperopt_search = HyperOptSearch()
optuna_search = OptunaSearch()
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


def process_stock(df):
    df = df[['date', 'ticker', 'closeadj']].drop_duplicates(['date', 'ticker'])
    df = pd.pivot(df, index='date', columns='ticker', values='closeadj').dropna(axis=1)
    return df


stock_mega = process_stock(stock_mega)
stock_large = process_stock(stock_large)
stock_mid = process_stock(stock_mid)
# read ETF data
ETF = pd.read_pickle('ETF.pkl').set_index('Date')[['Ticker', 'Close']]
ETF = ETF.loc[ETF.index >= '2010-01-01']
ETF = pd.pivot(ETF, columns='Ticker', values='Close')
ETF = delete_na_cols(ETF, 0.05).dropna(axis=0)
# data to use for testing
dataset = stock_mega
train_val_index = dataset.index[:int(len(dataset) * 0.6)]
test_index = dataset.index[int(len(dataset) * 0.6):]
train_val = dataset.loc[train_val_index]
test = dataset.loc[test_index]
# paramters for tune
param_grid = ParameterGrid(
    {
        'look_back': [0.5, 1, 1.5],
        'look_ahead': [0.5, 1, 1.5],
        'n_largest': [10, 20, 30],

        'hr': ['KF'],
        'OLS_window': [20],
        'constant': [False, True],

        'state_mean': [0, 0.1],
        'state_cov': [0.001, 0.005],
        'trans_var': [0.001, 0.005],
        'observ_var': [1, 2, 3],

        'signal': ['ecdf'],
        'signal_window': [20, 30, 60],
        'entry': [0.01, 0.025, 0.05],
        'exit': [-0.1, 0, 0.1]

    }
)

random_grid = {
    'look_back': tune.quniform(0.5, 2.5, 0.5),
    'look_ahead': tune.quniform(0.5, 2.5, 0.5),
    'n_largest': tune.qrandint(10, 40, 5),

    'hr_test': tune.choice(['1']),
    'hr_trade': tune.choice(['KF', 'OLS']),
    'OLS_window': tune.sample_from(lambda spec: 20 if spec.config.hr == 'KF' else tune.qrandint(20, 80, 10)),
    'constant': tune.choice([False, True]),

    'state_mean': tune.sample_from(lambda spec: tune.quniform(0, 0.15, 0.01) if spec.config.hr == 'KF' else 1),
    'state_cov': tune.sample_from(lambda spec: tune.quniform(0.001, 0.15, 0.001) if spec.config.hr == 'KF' else 1),
    'trans_var': tune.sample_from(lambda spec: tune.quniform(0.001, 0.15, 0.001) if spec.config.hr == 'KF' else 1),
    'observ_var': tune.sample_from(lambda spec: tune.quniform(1, 5, 0.1) if spec.config.hr == 'KF' else 1),

    'pvalue_zscore_weight': tune.quniform(0, 1, 0.1),

    'daily_reblance': tune.choice([False, True]),
    'signal': tune.choice(['ecdf', 'zscore']),
    'signal_window': tune.qrandint(20, 80, 10),
    'entry': tune.sample_from(
        lambda spec: tune.quniform(0.01, 0.1, 0.01) if spec.config.signal == 'ecdf' else tune.quniform(1.5, 3, 0.1)),
    'exit': tune.sample_from(
        lambda spec: tune.quniform(0.4, 0.6, 0.01) if spec.config.signal == 'ecdf' else tune.quniform(-1, 1, 0.1))
}

grid = {
    'look_back': tune.grid_search([0.5, 1]),
    'look_ahead': tune.grid_search([0.5, 1]),
    'n_largest': tune.grid_search([10, 20]),

    'hr': tune.grid_search(['KF']),
    'OLS_window': tune.grid_search([20]),
    'constant': tune.grid_search([False, True]),

    'state_mean': tune.grid_search([0, 0.1]),
    'state_cov': tune.grid_search([0.001]),
    'trans_var': tune.grid_search([0.001]),
    'observ_var': tune.grid_search([1]),

    'signal': tune.grid_search(['ecdf']),
    'signal_window': tune.grid_search([30]),
    'entry': tune.grid_search([0.05]),
    'exit': tune.grid_search([0])

}


# objective function
def objective(look_back, look_ahead, n_largest,
              hr_test, hr_trade, OLS_window, constant,
              state_mean, state_cov, trans_var, observ_var,
              daily_rebalance, signal, signal_window, entry, exit,
              pvalue_zscore_weight,
              data=train_val, n_job=1, save_perf=False, hide_progress_bar=True):
    rs = pd.Series(dtype='float64')
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    cutoff_date = min_date + datetime.timedelta(days=int(365 * look_back)) + datetime.timedelta(days=-signal_window)
    while True:
        # train test split
        train_start_date = cutoff_date + datetime.timedelta(days=-int(365 * look_back))
        val_end_date = cutoff_date + datetime.timedelta(days=int(365 * look_ahead))
        print(f'Training dates {train_start_date} - {val_end_date}')
        if cutoff_date <= max_date:
            dates_train = (data.index.date <= cutoff_date) & (data.index.date >= train_start_date)
            dates_val = (data.index.date > cutoff_date) & (data.index.date <= val_end_date)
            train, val = data.loc[dates_train], data[dates_val]
            # tune
            pt = PairsTrading(position_value=10000,
                              hr_test=hr_test, hr_trade=hr_trade, hr_window=OLS_window, hr_min_share_decimal=4,
                              hr_constant=constant,
                              perf_signal=signal, perf_signal_window=signal_window, perf_signal_entry=entry,
                              perf_signal_exit=exit,
                              perf_daily_rebalance=daily_rebalance,
                              n_jobs=n_job, backend='loky', hide_progress_bar=hide_progress_bar)
            trade1, perfs = memory.cache(pt.backtest_pairs)(data=train, pairs=None,
                                                            state_mean=state_mean, state_cov=state_cov,
                                                            trans_var=trans_var, observ_var=observ_var)
            perfs.to_csv('perfs.csv') if save_perf is True else None
            pairs = pt.select_pairs(perfs, weight=[pvalue_zscore_weight, 1 - pvalue_zscore_weight], nlargest=n_largest)
            trade2, perfs2 = pt.backtest_pairs(data=val, pairs=pairs)
            r = trade2['port_r']
            rs = rs.append(r)
            cutoff_date = val_end_date
        else:
            break
    # results
    result = {'look_back': look_back, 'look_ahead': look_ahead, 'n_largest': n_largest,
              'hr_test': hr_test, 'hr_trade': hr_trade, 'OLS_window': OLS_window, 'constant': constant,
              'state_mean': state_mean, 'state_cov': state_cov, 'trans_var': trans_var, 'observ_var': observ_var,
              'signal': signal, 'signal_window': signal_window, 'entry': entry, 'exit': exit,
              'sr': rs.mean() / rs.std() * sqrt(252) if ~((rs == 0).all()) else 0,
              'cumr': (1 + rs).cumprod().iloc[-1],
              'pairs': pairs,
              'rs': rs}
    return result


# trainning func
def training_function(config):
    tune.report(sr=objective(**config)['sr'])


# tune
now = datetime.datetime.now()
analysis = tune.run(
    training_function,
    num_samples=10000,
    # search_alg=optuna_search,
    verbose=0,
    metric='sr',
    mode='max',
    scheduler=ASHAScheduler(),
    resources_per_trial={'cpu': 1, 'gpu': 0},
    config=random_grid)

print('Best sr', analysis.best_result['sr'])
print("Best config: ", analysis.get_best_config())
print('total run time:', (datetime.datetime.now() - now).seconds)
# performance
for count, value in enumerate([train_val, test]):
    rs = objective(**random_grid, data=value, n_job=-1, save_perf=True)['rs']
    quantstats.reports.html(rs, 'SPY', output=f'report_{count}.html')

# my own parameters
my_param = {'look_back': 0.5,
            'look_ahead': 0.5,
            'n_largest': 30,
            'hr_test': 'KF',
            'hr_trade': 'KF',
            'OLS_window': 30,
            'constant': True,
            'state_mean': 0.09,
            'state_cov': 0.094,
            'trans_var': 0.058,
            'observ_var': 3,
            'pvalue_zscore_weight': 1,
            'daily_rebalance': True,
            'signal': 'zscore',
            'signal_window': 30,
            'entry': 2,
            'exit': 0}
rs = objective(**my_param, data=stock_mid, n_job=-1, save_perf=False, hide_progress_bar=False)['rs']
quantstats.reports.html(rs, 'SPY', output=f'report_my_param.html')


# single pair performance
def get_pair(X, y, look_back, look_ahead, n_largest,
             hr_test, hr_trade, OLS_window, constant,
             state_mean, state_cov, trans_var, observ_var, pvalue_zscore_weight,
             daily_rebalance, signal, signal_window, entry, exit, data=train_val, n_job=1):
    dfs = pd.DataFrame()
    pt = PairsTrading(position_value=10000,
                      hr_test=hr_test, hr_trade=hr_trade, hr_window=OLS_window, hr_min_share_decimal=4,
                      hr_constant=constant,
                      perf_signal=signal, perf_signal_window=signal_window, perf_signal_entry=entry,
                      perf_signal_exit=exit, perf_daily_rebalance=daily_rebalance,
                      n_jobs=n_job, backend='loky', hide_progress_bar=True)
    df = pt._backtest_pair(X, y, state_mean=state_mean, state_cov=state_cov, trans_var=trans_var, observ_var=observ_var)
    dfs = dfs.append(df)
    return dfs


params = my_param
dfs = get_pair(**params, X=ETF['SPY'], y=ETF['QQQ'])
dfs.to_csv('dfs.csv')
quantstats.reports.html(dfs['r_SPY_QQQ'], 'SPY', output=f'dfs.html')

# trade[['positions_short', 'positions_long']].plot()
# plt.show()
results = analysis.dataframe()
cols = ['sr'] + [i for i in results.columns if 'config' in i]
results = results[cols]
X = [i[7:] for i in results.columns if 'config' in i]
y = ['sr']
results.columns = y + X
results.to_pickle('results.pkl')
# graph to analysis
fig, ax = plt.subplots(
    round(len(X) / 3), 3, figsize=(15, 15), gridspec_kw={'hspace': 0.5, 'wspace': 0.25})
for i, ax in enumerate(fig.axes):
    if i < len(X) and X[i] != 'hr' and X[i] != 'signal':
        sns.regplot(
            x=X[i], y=y[0], data=results, ax=ax)
plt.savefig('res.png')
# regression analysis
hyper_res = pd.read_pickle('results.pkl')
hyper_res = pd.get_dummies(hyper_res, drop_first=True)
hyper_res['constant'] = hyper_res['constant'].astype(int)
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

y = hyper_res.pop('sr')
X = hyper_res
X = sm.add_constant(X)
ols_res = OLS(y, X.drop('constant', axis=1)).fit()
ols_res = pd.DataFrame({'params': ols_res.params, 'p_values': ols_res.pvalues})
