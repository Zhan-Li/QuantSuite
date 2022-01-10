# evaluate signal using stocks as investment
import pandas as pd
import pyspark.sql.functions as f
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window
from quantsuite  import DataManipulator

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
    .set("spark.executor.memory", '50g') \
    .set("spark.driver.memory", '50g') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
# .set("spark.sql.execution.arrow.pyspark.enabled", "true")\
# .set("spark.driver.maxResultSize", "32g") \
# .set("spark.memory.offHeap.size", "16g")
# .set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
# .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12") \
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# params
year = 2008
starting_date = f'{year}-01-01'
ending_date = f'{year}-12-31'
daily_money = 2000
zscore_start = -20
# read option data and stock data
opt = spark.read.parquet('data/opprcd_1996_2019') \
    .filter(f.col('date') >= starting_date) \
    .filter(f.col('date') <= ending_date) \
    .drop('ticker')
opt = DataManipulator().prc_to_r(opt, 'optionid', 'date', 'opt_prc', 'opt_r', fee=0)
opt = DataManipulator().r_to_cumr(opt, 'optionid', 'date', 'opt_r', 1, 30, 'opt_forward_r_30')
opt = DataManipulator().r_to_cumr(opt, 'optionid', 'date', 'opt_r', 1, 1, 'opt_forward_r_1')
opt = DataManipulator().r_to_cumr(opt, 'optionid', 'date', 'opt_r', 1, Window.unboundedFollowing, 'opt_forward_r_exp')
opt = DataManipulator().add_timeseries_zscore(opt, 'cusip', 'date', 'impl_volatility', zscore_start, 0, 'IV_zscore')

sec = spark.read.parquet('data/dsf_linked') \
    .filter(f.col('date') >= starting_date) \
    .filter(f.col('date') <= ending_date) \
    .select('date', 'cusip', 'ticker', 'prc', 'vol', 'mktcap', 'siccd', 'ret')

# sample high tech sic code
tickers = ['APPL', 'MSFT', 'TSLA', 'NVDA', 'FB', 'GOOG', 'SQ', 'NET', 'LYFT', 'DKNG', 'CVNA', 'LAZR', 'AMD']
siccodes = []
for ticker in tickers:
    try:
        sic = sec.filter(f.col('ticker') == ticker).select('siccd').distinct().collect()[0]['siccd']
        siccodes.append(sic)
    except Exception as e:
        print(e)
# merge
merged = opt.join(sec, on=['date', 'cusip']) \
    .filter(f.col('siccd').isin(siccodes)).cache()
merged = opt.join(sec, on=['date', 'cusip']).cache()
# which options to invest?
min_exp_days = 31
min_prc_pct = 0.2  # minimum percentage difference between stock price and strike
window = Window.partitionBy('date', 'cusip', 'cp_flag')
select_exdays = merged \
    .filter(f.col('exdays') >= min_exp_days) \
    .withColumn('min_exp_days', f.min('exdays').over(window)) \
    .filter(f.col('exdays') == f.col('min_exp_days'))
select_puts = select_exdays.withColumn('max_put_strike', f.col('prc') * (1 - min_prc_pct)) \
    .filter(f.col('strike_price') <= f.col('max_put_strike')) \
    .withColumn('max_put_strike', f.max('strike_price').over(window)) \
    .filter(f.col('strike_price') == f.col('max_put_strike')) \
    .filter(f.col('cp_flag') == 'P') \
    .drop('max_put_strike')
select_calls = select_exdays.withColumn('min_call_strike', f.col('prc') * (1 + min_prc_pct)) \
    .filter(f.col('strike_price') >= f.col('min_call_strike')) \
    .withColumn('min_call_strike', f.min('strike_price').over(window)) \
    .filter(f.col('strike_price') == f.col('min_call_strike')) \
    .filter(f.col('cp_flag') == 'C') \
    .drop('min_call_strike')
data = select_puts.unionByName(select_calls).cache()

# option selection
r = 'opt_forward_r_30'
IV_zscore_min = 0.9
IV_min = 0
data = data.filter(f.col('IV_zscore') >= IV_zscore_min) \
    .filter(f.col('impl_volatility') >= IV_min) \
    .filter(f.col('opt_vold') > 10000) \
    .filter(f.col('ret') < 0) \
    .filter(f.col(r).isNotNull()) \
    .withColumn('position_EW', f.lit(-1)) \
    .withColumn('count', f.count(r).over(Window.partitionBy('date'))) \
    .withColumn('size_per_option', -daily_money / f.col('count')) \
    .withColumn('position_count', -f.col('count')) \
    .withColumn('IV_inverse', 1 / f.col('impl_volatility')) \
    .withColumn('position_IV', -f.col('IV_inverse') / f.sum('IV_inverse').over(Window.partitionBy('date'))) \
    .cache()

# get portfolio return

portr = data \
    .groupby('date').agg(f.mean(f.col(r) * f.col('position_EW')).alias('port_r')).sort('date').toPandas()
portr.mean()
portr.min()
