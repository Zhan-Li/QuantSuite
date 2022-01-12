from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from quantsuite.signals import TechnicalSignal
from quantsuite import DataManipulator
from sqlalchemy import create_engine
import json
import pandas as pd

spark_conf = SparkConf() \
    .set("spark.executor.memory", '50g') \
    .set("spark.driver.memory", '50g') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

# get data
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin = secrets['mysql_password']
start = '1996-01-01'
crsp_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/crsp')
pd.read_sql(f"SELECT * FROM stock WHERE date >='{start}'", con=crsp_connection)\
    .sort_values('date')\
    .to_parquet('crsp.parquet')
# read data
stock = spark.read.parquet('crsp.parquet').filter(f.col('date') >= '2006-01-01')
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

stock_sigs = DataManipulator(stock_sigs, 'date', 'cusip') \
    .add_mean('turnover_daily', -126, 0, 'turnover_6m_avg') \
    .add_rsd('turnover_daily', -126, 0, 'turnover_6m_rsd') \
    .add_mean('vol_d', -126, 0, 'vol_d_6m_avg') \
    .add_rsd('vol_d', -126, 0, 'vol_d_6m_rsd') \
    .add_mean('illiq_daily', -126, 0, 'illiq_6m_avg') \
    .add_skewness('ret', -21, 0, 'skewness_1m') \
    .add_kurtosis('ret', -21, 0, 'kurtosis_1m') \
    .add_mean('spread_daily', -21, 0, 'spread_1m').data

