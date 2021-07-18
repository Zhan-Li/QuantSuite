# evaluate signal using stocks as investment
# failed with a 0.8 sharpe ratio in the training period.Cannot beat SPY.
import pandas as pd
import pyspark.sql.functions as f
import importlib
from sig_option import OptionSignal
import sig_option
importlib.reload(sig_option)
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortforlioReturn, ReturnAnalysis
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
starting_date = '2011-01-01'
ending_date = '2020-12-31'
forward_r = 'forward_r'
sig_name = 'n_r'
# read stock data
sec = spark.read.parquet('data/dsf_linked') \
    .filter(f.col('date') >= starting_date)\
    .filter(f.col('date') <= ending_date)\
    .withColumn(forward_r, f.lead('ret').over(Window.partitionBy('cusip').orderBy('date')))\
    .select('date', 'cusip', 'ticker', 'vol', 'vol_d', 'prc', 'ret', forward_r).cache()
# create signal
def objective(volc_min, ntile=10, report=False):
    sig = TechnicalSignal(sec, 'date', 'cusip')
    data = sig.add_consecutive_r('ret', sig_name).data\
            .filter(f.col('vol_d') >= volc_min)
    # portfolio analyais
    r = PortforlioReturn(data, 'date', 'cusip', sig=sig_name, forward_r= forward_r)\
            .univariate_portfolio_r(ntile=ntile).toPandas().sort_values('date')
    r_selected = r.loc[r['sig_rank'] == 'high_minus_low'][forward_r]
    sr = np.abs(r_selected.mean()/r_selected.std()*np.sqrt(252))
    if report is  True:
        r_hl = r.loc[r['sig_rank'] == 'high_minus_low'].set_index('date')[forward_r]
        quantstats.reports.html(-r_hl, 'SPY', output = 'report.html')
        ra = ReturnAnalysis(r, 'date', forward_r, 'sig_rank', 1)
        print(ra.sort_return())
    if sr == np.nan:
        return 0
    if sr != np.nan:
        return sr

def optimize(trial):
    volc_min = trial.suggest_discrete_uniform('volc_min', 100000, 10000000, 10000)
    return  objective(volc_min, ntile=10)

study = optuna.create_study(direction="maximize")
study.optimize(optimize,  n_jobs=-1, n_trials=1000)
joblib.dump(study, 'consecutive_r.pkl')
# read optimized result
study_file = 'consecutive_r.pkl'
mystudy = joblib.load(study_file)
best_params = mystudy.best_trial.params
test = sec
# retest
objective(480000, report=True)