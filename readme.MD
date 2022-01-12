<h1 align="center">QuantSuite</h1>
<p align="center">
QuantSuite is a suite of tools for quantitative research built on PySpark, Ray Tune, Scikit-Learn,
XGBoost, and TensforFlow.
</p>

```diff
! This is a private respository but made public recently. No documentation will be provided. 
```

<h2> Key Features </h2>
<ul>
<li>quantsuite.data_processor: Data process for WRDS databases</li>
<li>quantsuite.downloader: Financial data downloader</li>
<li>quantsuite.signals: Extract stock trading signals</li>
<li>quantsuite.forcaster: Costomized hyperparameter searching and return prediction </li>
<li>quantsuite.portfolio_analysis: Portfolio analysis and performance evaluation</li>
<li>quantsuite.trader: Trading bots for Binance and Interactive Brokers</li>
</ul>

<h2> Example Usage </h2>
<h3>Create technical signals from stock data</h3>

```python
from quantsuite.signals import TechnicalSignal
stock = spark.read.parquet('crsp.parquet')
stock_sigs = TechnicalSignal(stock, 'date', 'cusip') \
    .add_cumr('ret', -252, -21, 'r11m') \
    .add_cumr('ret', -147, -21, 'r6m') \
    .add_cumr('ret', -21, 0, 'r1m') \
    .add_52w_high('prc_adj', 'high_52w') \
    .add_std('ret', -21, 0, 'r_std') \
    .add_avg_r('ret', -21, 0, 'r_avg') \
    .add_turnover('vol', 'shrout', 'turnover_daily') \
    .add_illiq('ret', 'vol_d', 'illiq_daily') \
    .add_max_min_r('ret', -21, 0, 'max_r_1m', 'min_r_1m') \
    .add_drawdown('ret', 'drawdown')\
    .add_spread('bid', 'ask', 'spread_daily').data
```

<h3>Stock Return Prediction</h3>
Predict Stock Returns with Machine Learning

```python
from quantsuite.forcaster import Forecaster, CustomCV, TrainTestSplitter, Pipe
# train test split based on a cut-off date
train, test = TrainTestSplitter(stock_data).split(test_offset_periods = -12, train_end_time=f'2019-12-31')
x_train, y_train = train.drop(['target'], axis=1), train['target']
x_test, y_test = test.drop(['target'], axis=1), test['target']
# Custom cross-validation index, keeping temporal order
cv_train = CustomCV(train).gen_cv_index(train_window_size=12, val_window_size =12, window_size =12,
                                        moving_window=True)
# build pipeline
num_cols = x_train.iloc[:, :-1].columns
# Pipeline with pre-built search space
pipe = Pipe('random_forest_regressor', num_cols)
model = pipe.build_pipline()
params = pipe.get_params()
rf = Forecaster(x_train, y_train, cv_train)
# Hyperparameter search with information coefficient as the evaluation metric
rf.search(params, model, scoring = 'IC', n_trial=1000, max_search_seconds=2 * 60 * 60, 
          n_jobs = -1, use_gpu = False)
# predict with the best hyperparameter
rf.predict(x_test, y_test, train_pred_index, 'date', 'daily')
```

<h3>Performance Evaluation</h3>
Evaluating signal performance with portfolio sorts

```python
from quantsuite import PortfolioAnalysis
pa = PortfolioAnalysis(stock_data, time='date', name = 'cusip', sig='my_signal', forward_r='forward_r')
# Decile single sort
returns = pa.gen_portr(shift=1, ntile=10)
# 10X10 independent double sort
returns = pa.gen_portr(shift=1, ntile=10, sort_var2='mtk_cap', ntile2=10, dependent_sort=False)
# 10X10 dependent double sort
returns = pa.gen_portr(shift=1, ntile=10, sort_var2='mtk_cap', ntile2=10, dependent_sort=True)
```

<h2>A Sample Result </h2>
Backtesting results with looking-ahead bias carefully removed. Plot was generateed in 
<a href = "https://github.com/ranaroussi/quantstats">QuantStats </a>.
<ul>
<li>Model: Random Forest Regressor </li>
<li>Sample Period: 2007 - 2020 </li>
<li>Universe: Optionable US stocks </li>
<li>Signals: Extracted from OptionMetrics </li>
<li>Balance Schedule: Monthly </li>
<li>Transaction Costs: Not accounted for </li>
<li>Number of hyperparameter samples: 100 </li>
<li>Look back periods: 12 months </li>
</ul>
<img src = "https://github.com/Zhan-Li/QuantSuite/blob/711f7c4d8c8c76b596a118e3ecef835fd177443c/sample_results.png" alt = "">
