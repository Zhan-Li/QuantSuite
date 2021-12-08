# evaluate signal using stocks as investment
# failed signal with maximized sharpe ratio at 0.6
import pandas as pd
import pyspark.sql.functions as f
import importlib
from sig_option import OptionSignal
import sig_option
importlib.reload(sig_option)
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortforlioAnalysis
import sig_AMFE; importlib.reload(sig_AMFE)
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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
        #.set("spark.sql.execution.arrow.pyspark.enabled", "true")\
        #.set("spark.driver.maxResultSize", "32g") \
        #.set("spark.memory.offHeap.size", "16g")
        #.set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
        #.set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12") \
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# fixed parameters
starting_date = '2007-01-01'
ending_date = '2016-12-31'
sig_col = 'prc_ratio'
forward_r = 'forward_r'
# generate signal
sec = spark.read.parquet('data/dsf_linked')\
    .filter(f.col('date') >= starting_date)\
    .filter(f.col('date') <= ending_date)\
    .withColumn(forward_r, f.lead('ret').over(Window.partitionBy('cusip').orderBy('date'))).cache()
opt = spark.read.parquet('data/opprcd_1996_2020')\
    .filter(f.col('date') >= starting_date)\
    .filter(f.col('date') <= ending_date) \
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer'))/2)\
    .withColumn('dummy', f.when(f.col('cp_flag') == 'C', 1).when(f.col('cp_flag') == 'P', -1))\
    .withColumn('open_interest', f.col('open_interest')*f.col('dummy'))\
    .withColumn('opt_vold', f.col('opt_vold')*f.col('dummy')) \
    .withColumn('oi_adj_change', f.col('oi_adj_change')*f.col('dummy'))
# create signal
def objective(weight, volc_min, ntile):
    data = OptionSignal(opt, 'date', 'cusip', 'cp_flag') \
        .add_opt_stk_sig_ratio('opt_prc', weight, False, sec, 'prc', sig_col).data\
        .filter(f.col('vol_d') >= volc_min)
    # portfolio analyais
    pa = PortforlioAnalysis(data, 'date', 'cusip', sig=sig_col, forward_r= forward_r)
    r = pa.univariate_portfolio_r(ntile=ntile).toPandas().sort_values('date')
    r_h= r.loc[r['sig_rank'] == str(ntile)][forward_r]
    sr = r_h.mean()/r_h.std()*np.sqrt(252)
    if sr == np.nan:
        return 0
    if sr != np.nan:
        return np.abs(sr)

def optimize(trial):
    weight = trial.suggest_categorical('weight', ['volume', 'open_interest', 'opt_vold', 'oi_adj_change'])
    #volc_min = trial.suggest_discrete_uniform('volc_min', 100000, 100000, 10000)
    volc_min = 100000
    ntile = 10
    return  objective(weight, volc_min, ntile)

study = optuna.create_study(direction="maximize")
study.optimize(optimize, n_jobs=-1, n_trials=1000)
joblib.dump(study, 'stk_opt_prc_ratio.pkl')
# failure with highest sharpe ratio at about 0.6

# read optimized result
study_file = 'vol.pkl'
mystudy = joblib.load(study_file)
study_df = mystudy.trials_dataframe().sort_values('value', ascending=False)
best_params = mystudy.best_trial.params
# test params
volc_min = best_params['volc_min']
look_back = best_params['look_back']
ntile = best_params['ntile']
return_min = best_params['return_min']
test = sec
# retest
test_opt = opt.filter(f.col('date') >='2020-01-01')\
    .withColumn('opt_vol', when())
sig = OptionSignal(opt, sec, 'date', 'cusip', 'cp_flag')
data = sig.aggregate_sig('opt_vol', weight=None,)
data = data.filter(f.col('vol_c') >= volc_min)\
        .filter(f.abs(f.col('ret')) <= return_min)
pa = PortforlioAnalysis(data, 'date', 'ticker', sig=sig_col, forward_r= forward_r)
r = pa.univariate_portfolio_r(ntile=ntile).toPandas().sort_values('date')
r = r.set_index('date')
pa.sort_return(r, days=2)
r_h = r.loc[r['sig_rank'] == str(ntile)][forward_r][::2]
r_h.mean()/r_h.std()*np.sqrt(252/2)
quantstats.reports.html(r_h, 'SPY', output='reporth.html')
r_hl = r.loc[r['sig_rank'] == 'high_minus_low'][forward_r][::2]
r_hl.mean()/r_hl.std()*np.sqrt(252)
quantstats.reports.html(r_hl, 'SPY', output='report.html')