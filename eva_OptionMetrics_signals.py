# evaluate signal using stocks as investment
# bas performance with 0.61 SR in the 15-20 test period
# generally, when aggregating option level data to stock level data, value to be aggregated should be positive for call
# and nagetive for put, and then weight need to be positive.
import pandas as pd
import pyspark.sql.functions as f
from pyspark import StorageLevel
import importlib
import sig_option
importlib.reload(sig_option)
from sig_option import OptionSignal

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortfolioAnalysis
import utilities as utils
import quantstats
import sig_technical; importlib.reload(sig_technical)
import numpy as np
from sig_technical import TechnicalSignal
import optuna
import joblib
import data_manipulator
importlib.reload(data_manipulator)
from data_manipulator import DataManipulator
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")\
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf) \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/opt_sig.sorting_results") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/opt_sig.sorting_results") \
    .getOrCreate()

#people.write.format("mongo").mode("append").save()


# fixed parameters
train_start = '2011-01-01'
train_end = '2020-12-31'
forward_r = 'forward_r'
vold_min = 200000
vars = ['vol', 'prc', 'ret', 'mktcap', 'shrout']
sig_name = 'os'
# option data
sec = spark.read.parquet('data/dsf_linked')
opt = spark.read.parquet('data/opprcd_1996_2020') \
    .filter(f.col('date') >= train_start) \
    .filter(f.col('date') <= train_end) \
    .join(sec, on=['date', 'cusip'])\
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('ks', f.col('strike_price')/f.col('prc')) \
    .withColumn('ks>1?', f.when(f.col('ks') >1, True).when(f.col('ks') <= 1, False))\
    .withColumn('ks+-', f.when(f.col('cp_flag') == 'P', -1*f.col('ks')).otherwise(f.col('ks'))) \
    .withColumn('IV+-', f.when(f.col('cp_flag') == 'P', -1 * f.col('impl_volatility')).otherwise(f.col('impl_volatility'))) \
    .withColumn('exdays+-',f.when(f.col('cp_flag') == 'P', -1*f.col('exdays')).otherwise(f.col('exdays')))\
    .withColumn('elasticity', f.abs('delta') * f.col('prc') / f.col('opt_prc'))\
    .withColumn('elasticity+-',f.when(f.col('cp_flag') == 'P', -1*f.col('elasticity')).otherwise(f.col('elasticity')))\
    .withColumn('opt_vol_adj+-', f.when(f.col('cp_flag') == 'P', -1*f.col('opt_vol_adj')).otherwise(f.col('opt_vol_adj')))\
    .withColumn('opt_vold+-', f.when(f.col('cp_flag') == 'P', -1*f.col('opt_vold')).otherwise(f.col('opt_vold'))) \
    .withColumn('open_interest_adj+-',f.when(f.col('cp_flag') == 'P', -1 * f.col('open_interest_adj')).otherwise(f.col('open_interest_adj')))\
    .withColumn('oi_adj_change+-', f.when(f.col('cp_flag') == 'P', -1*f.col('oi_adj_change')).otherwise(f.col('oi_adj_change')))
# note that oi_adj_change is not suitable to act weight as it have both negative and positive values.
# option signals
## os signal
os_sigs = []
for opt_vol in ['opt_vol_adj+-', 'opt_vold+-', 'open_interest_adj+-', 'oi_adj_change+-']:
    for sec_vol in ['vol_adj', 'vol_d']:
        for weight in [None, 'elasticity']:
            opt = opt.withColumn('os', f.col(opt_vol)/f.col(sec_vol))
            os_sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P')\
                .gen_os('os', weight, f'os|{opt_vol}|{sec_vol}|{weight}')
            os_sigs.append(os_sig)
os = utils.multi_join(os_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## opt volume signal]
vol_sigs = []
for opt_vol in ['opt_vol_adj+-', 'opt_vold+-', 'oi_adj_change+-']:
    for weight in [None, 'elasticity']:
        vol_sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
            .gen_signed_vol(opt_vol, weight, f'vol|{opt_vol}|{weight}')\
            .withColumn(f'CP_ratio|{opt_vol}|{weight}', f.col(f'vol|{opt_vol}|{weight}_C')/f.abs(f'vol|{opt_vol}|{weight}_P'))
        vol_sig = DataManipulator(vol_sig, 'date', 'cusip')\
            .add_timeseries_comparison(f'vol|{opt_vol}|{weight}_C', -20, 0, f'z|{opt_vol}|{weight}_C', f'ratio|{opt_vol}|{weight}_C')\
            .add_timeseries_comparison(f'vol|{opt_vol}|{weight}_P', -20, 0, f'z|{opt_vol}|{weight}_P', f'ratio|{opt_vol}|{weight}_P')\
            .add_timeseries_comparison(f'vol|{opt_vol}|{weight}_CP', -20, 0, f'z|{opt_vol}|{weight}_CP', f'ratio|{opt_vol}|{weight}_CP')\
            .data
        vol_sigs.append(vol_sig)
vol = utils.multi_join(vol_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## leverage measures
lvg_sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    lvg_sig1 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
        .gen_ks('ks+-', weight, f'ks|{weight}')
    lvg_sig2 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
        .gen_expdays('exdays+-', weight, f'expdays|{weight}')
    lvg_sig3 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
        .gen_elasticity('elasticity+-', weight, f'elasticity|{weight}')
    lvg_sig = utils.multi_join([lvg_sig1, lvg_sig2, lvg_sig3], on=['date', 'cusip'])
    lvg_sigs.append(lvg_sig)
lvg = utils.multi_join(lvg_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## IV spread
IV_spread_sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    IV_spread_sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
        .gen_IV_spread('IV+-', weight, f'IV_spread|{weight}')
    IV_spread_sigs.append(IV_spread_sig)
IV_spread = utils.multi_join(IV_spread_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## IV skew
IV_skew_sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    IV_skew_sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
        .gen_IV_skew('IV+-', weight, 'ks>1?', f'IV_skew|{weight}')
    IV_skew_sigs.append(IV_skew_sig)
IV_skew = utils.multi_join(IV_skew_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## all signals
signals = IV_skew
# stock return data
sec = spark.read.parquet('data/dsf_linked') \
    .filter(f.col('vol_d') > vold_min) \
    .withColumn(forward_r, f.lead('ret').over(Window.partitionBy('cusip').orderBy('date'))) \
    .select('date', 'cusip', 'ticker', forward_r, 'vol', 'vol_adj', 'vol_d', 'prc', 'ret', 'mktcap', 'shrout', 'ret_SPY')
# merge stock return with option signal
merged = sec.join(signals, on=['date', 'cusip']).cache()
# Exploration
results = pd.DataFrame()
for sig_col in signals.columns[2:]:
    pa = PortfolioAnalysis(merged, 'date', 'cusip', sig=sig_col, forward_r=forward_r, var_list=vars)
    returns = pa.gen_portr(ntile=10, total_position=0, commission=0, turnover=0)
    sorted = pa.sort_return(days=1, trading_cost=False)
    sorted['sig_name'] = sig_col
    sorted['result_time'] = datetime.datetime.now()
    results = results.append(sorted)
    r = returns.loc[returns['sig_rank'] =='high_minus_low'][forward_r]
results.to_csv('opt_sig_results.csv', mode='a', header=False)
# plot
quantstats.reports.html(r, 'SPY', output=f'report_{sig_col}.html')

# code requires change -------------------------------
# variable params
sig_name = 'os'
output_file = 'os.pkl'
# objective function
def objective(opt_vol, sec_vol, weight, suffix, n, days, sig_rank_selected='high_minus_low',
              report=False, filename='report.html', low_minus_high=False, trading_cost=True, absolute_SR=True):
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', ['C', 'P']) \
        .gen_os(opt_vol, weight, sec, sec_vol,  sig_name)
    train = sec.join(sig, on=['date', 'cusip']).filter(f.col('date') <= train_end)
    data = DataManipulator(train, 'date', 'cusip') \
        .add_avg(sig_name+suffix, -(days - 1), 0, sig_name+suffix) \
        .add_cumr('ret', 1, days, forward_r) \
        .resample(days) \
        .data
    pa = PortfolioAnalysis(data, 'date', 'cusip', sig=sig_name+suffix, forward_r=forward_r, var_list=vars)
    returns = pa.gen_portr(n=n, total_position=total_position, commission=0, turnover=1)

    r_selected = returns.loc[returns['sig_rank'] == sig_rank_selected][::days]
    r_cost = r_selected['trading_cost'] if trading_cost is True else 0
    r_raw = r_selected[forward_r]
    r = r_raw - r_cost
    sr = r.mean()/r.std()*np.sqrt(252/days)
    sr = np.abs(sr) if absolute_SR is True else sr
    if report is True:
        if low_minus_high is True:
            r = -r
        quantstats.reports.html(r, 'SPY', output=filename)
        print(pa.sort_return(days=days, trading_cost=trading_cost))
        return returns
    else:
        if sr == np.nan:
            return 0
        if sr != np.nan:
            return sr

# optimization

def optimize(trial):
    opt_vol = trial.suggest_categorical('opt_vol', ['opt_vol_adj', 'opt_vold', 'oi_adj_change'] )
    sec_vol = trial.suggest_categorical('sec_vol', ['vol_adj', 'vol_d'])
    weight = trial.suggest_categorical('weight', ['None', 'elasticity_abs'])
    suffix = trial.suggest_categorical('suffix', ['_C', '_P', '_CP'])
    n = trial.suggest_int('n', 1, 100, 10)
    days = trial.suggest_int('days', 1, 5, 1)
    return objective(opt_vol, sec_vol, weight, suffix, n,  days, sig_rank_selected='high_minus_low', absolute_SR=True)

# search for parameter -------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(optimize,  n_jobs=-1, n_trials=10000)
study_df = study.trials_dataframe()
joblib.dump(study, output_file)

# read optimized result
study_file = output_file
mystudy = joblib.load(study_file)
best_params = mystudy.best_trial.params
print(best_params)
