# evaluate signal using stocks as investment
import pandas as pd
import pyspark.sql.functions as f
import importlib
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortforlioAnalysis
import sig_AMFE; importlib.reload(sig_AMFE)
import data_manipulator; importlib.reload(data_manipulator)
from data_manipulator import DataManipulator
import sig_option
importlib.reload(sig_option)
from sig_option import OptionSignal
import performance_analysis
importlib.reload(performance_analysis)
from performance_analysis import PerformanceAnalytics
from sig_AMFE import AMFE
import data_manipulator
import utilities as utils
import quantstats
import sig_technical; importlib.reload(sig_technical)
import numpy as np
from sig_technical import TechnicalSignal
from pyspark.sql.types import StringType
import optuna
import joblib
from tpot import TPOTRegressor
import option_selector
importlib.reload(option_selector)
from option_selector import OptionSelector
import pyfolio
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .set('spark.driver.maxResultSize', '1g')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# fixed parameters
train_start_date = '2010-01-01'
train_end_date = '2020-12-31'
# read crypto data
crypto = spark.createDataFrame(pd.read_pickle('data/crypto_1d.pkl'))\
    .withColumn('close_price_lagged', f.lag('close').over(Window.partitionBy('symbol').orderBy('close_time')))\
    .withColumn('r', f.col('close')/f.col('close_price_lagged') - 1)\
    .withColumn('forwardr', f.lead('r').over(Window.partitionBy('symbol').orderBy('close_time')))\
    .withColumn('ret_avg',
                (f.abs(f.col('open')/f.col('close_price_lagged') - 1) +
                f.abs(f.col('high')/f.col('close_price_lagged') - 1) +
                f.abs(f.col('low')/f.col('close_price_lagged') - 1) +
                f.abs('r'))/4)
# training params
forward_r = 'forwardr'
sig_name = 'vol_ratio'
time = 'close_time'
train= crypto
test = crypto
# create signal
def objective(df, vold_min, look_back, return_min, n, every, report):
    df = TechnicalSignal(df, time, 'symbol')\
        .add_value_average_ratio('volume', look_back, -1, sig_name)\
        .data\
        .filter(f.col('quote_asset_volume') >= vold_min)\
        .filter(f.abs('ret_avg') <= return_min)
    pa = PortforlioAnalysis(df, time, 'symbol', sig=sig_name, forward_r= forward_r)
    r = pa.univariate_portfolio_r2(n=n).toPandas().sort_values(time).set_index(time)
    r_h = r.loc[r['sig_rank'] == 'high'][forward_r][::every]
    r_hl = r.loc[r['sig_rank'] == 'high_minus_low'][forward_r][::every]
    # try:
    #     fees = pa.get_transact_cost(r, 1, total_position)
    #     fees = fees.set_index('date')
    # except Exception as e:
    #     print(e)
    #     fees = {}
    #     fees['pct_fee_l'] = r['n_assets'].mean()*0.9*2/total_position

    sr = r_hl.mean()/r_hl.std()*np.sqrt(252/every)
    if report is True:
        print(pa.sort_return(r, days=every))
        quantstats.reports.html(r_hl, 'SPY', output='reporth.html')
    return sr if sr != np.nan else 0

def optimize(trial):
    volc_min = 10000
    look_back = trial.suggest_int('look_back', -100, -1, 1)
    return_min = trial.suggest_discrete_uniform('return_min', 0.001, 0.5, 0.001)
    n = trial.suggest_int('n', 1, 100, 1)
    return objective(train, volc_min, look_back, return_min, n, every=1, report=False)

study = optuna.create_study(direction="maximize")
study.optimize(optimize, n_jobs=-1, n_trials=5000)
joblib.dump(study, f'crypo_vol_ratio.pkl')

# read optimized result
# best results:
#  -69,
#  33,
#  0.036000000000000004,
#  'COMPLETE']
study_file = 'crypo_vol_ratio.pkl'
mystudy = joblib.load(study_file)
study_df = mystudy.trials_dataframe().sort_values('value', ascending=False)
best_result = mystudy.best_trial.params
# test params
vold_min = 100000
df = TechnicalSignal(test, time, 'symbol') \
    .add_value_average_ratio('volume', best_result['look_back'], -1, sig_name) \
    .data \
    .filter(f.col('quote_asset_volume') >= vold_min) \
    .filter(f.abs('ret_avg') <= best_result['return_min'])
pa = PortforlioAnalysis(df, time, 'symbol', sig=sig_name, forward_r=forward_r)
r = pa.univariate_portfolio_r2(n=best_result['n']).toPandas().sort_values(time)
r = r.loc[r['forwardr'].notnull()]
print(pa.sort_return(r, days=1))
PA = PerformanceAnalytics(r.loc[r['sig_rank'] == 'high'], 'close_time', 'forwardr')
PA.plot_wealth()


# machinelarning with TPOT

# machine learning
df = TechnicalSignal(sec, 'cusip', 'date') \
    .add_value_average_ratio('vol', look_back, -1, 'vol_ratio').data\
    .select('date', 'cusip', 'vol_ratio', 'vol_d', 'ret', 'ret_o', 'ret_l', 'ret_h', 'ret_avg', 'forward_r2')\
    .filter(f.col('vol_d')>= 100000)\
    .dropna() \
    .toPandas().sort_values('date').reset_index()

def customize_cv(df, time: str, window: int):
    groups = df.groupby(df[time].dt.date).groups
    sorted_groups = [value for (key, value) in sorted(groups.items())]
    return [(sorted_groups[i], sorted_groups[i+1]) for i in range(window-1, len(sorted_groups)-1)]

cv = customize_cv(df, 'date', 1000)

X_train, y_train = df[['vol_ratio', 'ret_avg']], df[['forward_r2']]

# prediction
tpot = TPOTRegressor(generations=100, population_size=100, scoring = 'neg_mean_squared_error',  verbosity=2, cv = cv,
                     max_eval_time_mins=10,  n_jobs=-1, early_stop=10, memory = 'auto', random_state = 0)
tpot.fit(X_train.values, y_train.values.reshape(-1,))
y_pred = tpot.predict(X_train)
tpot.score(X_train, y_train)
tpot.export('stock_vol_signal_models.py')
tpot.fitted_pipeline_
# evaluation
sig = 'y_sig'
df['y_sig'] = y_pred
df.to_parquet('df.parquet')
my_df = spark.read.parquet('df.parquet').filter(f.col('vol_d')>= 100000)
pa = PortforlioAnalysis(my_df, 'date', 'cusip', sig, 'forward_r2')
r = pa.univariate_portfolio_r2(n).toPandas().sort_values('date').set_index('date')
print(pa.sort_return(r, days=2))
r_hl = r.loc[r['sig_rank'] == 'high_minus_low']['forward_r2'][::2]
quantstats.reports.html(r_hl, 'SPY', output='report2.html')
