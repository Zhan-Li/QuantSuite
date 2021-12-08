# evaluate signal using stocks as investment
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
import sig_technical_time_series
importlib.reload(sig_technical_time_series)
from sig_technical_time_series import TimeSeriesTechnicalSignal

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
train_start_date = '2011-01-01'
ending_date = '2020-12-31'
forward_r = 'forward_r'
# read stock data
sec = spark.read.parquet('data/dsf_linked')\
    .filter((f.col('shrcd') == 10)|(f.col('shrcd') == 11) |(f.col('shrcd') == 12)|(f.col('shrcd') == 18)|
            (f.col('shrcd') == 30)|(f.col('shrcd') == 31))\
    .filter(f.col('date') >= train_start_date)
sec = data_manipulator.r_to_cumr(sec, 'cusip', 'date', 'ret', 1, 1, 'forward_r')\
    .select('date', 'cusip', 'ticker', 'vol', 'vol_c', 'prc', 'ret', 'forward_r')


sig = TimeSeriesTechnicalSignal(sec, 'date', 'cusip')
data = sig.gen_mkt_breadth('ret', 'breadth')\
    .withColumn('breadth_lag', f.lag('breadth').over(Window.orderBy('date')))\
    .withColumn('breadth_diff', f.col('breadth')-f.col('breadth_lag'))\
    .drop('breadth_lag').cache()
# get SPY data
SPY = utils.download_return(start_date=train_start_date, end_date='2021-01-01', symbol='SPY')
SPY = spark.createDataFrame(SPY)\
    .select(f.col('Date').alias('date'), f.col('r').alias('r_SPY'))\
    .withColumn('ticker', f.lit('SPY'))
SPY = data_manipulator.r_to_cumr(SPY, 'ticker', 'date', 'r_SPY', 1, 1, 'forward_r_SPY')\
    .select('date', 'r_SPY', 'ticker' ,'forward_r_SPY')
data = data.join(SPY, on='date')
data = data_manipulator.add_timeseries_zscore(data, 'ticker', 'date', 'breadth', -10, 0, 'breadth_zscore').cache()

#
cutoff = 0.2
sig = 'breadth_diff'
result = data\
    .withColumn('trade', f.when(f.col(sig)>=cutoff,1).when(f.col(sig)<cutoff, -1).otherwise(0))\
    .withColumn('port_r', f.col('forward_r_SPY')*f.col('trade'))

r= result.select('date', 'port_r').toPandas().sort_values('date').set_index('date')
quantstats.reports.html(r['port_r'], 'SPY', output='report.html')
