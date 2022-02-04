import json

import pandas as pd
from binance.client import Client
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

import quantsuite.trader.binance as binance_funcs
from quantsuite.signals import TechnicalSignal
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# params
numOfAssets = 15
percentInvest = 0.95
# spark
spark_conf = SparkConf() \
    .set("spark.executor.memory", '30g') \
    .set("spark.driver.memory", "30g") \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# binance API
with open('secret.json') as f:
    secret = json.load(f)
api_key = secret['binance_api_key']
api_secret = secret['binance_api_secret']
client = Client(api_key, api_secret)
# current position
isolated_margin_account = binance_funcs.get_isolated_margin_account(client, 'USDT')
isolated_margin_account = isolated_margin_account.loc[isolated_margin_account['symbol'] == 'BTCUSDT']
currentPosition = isolated_margin_account[['symbol', 'base_netAssetOfQuote']]
totalMoney = isolated_margin_account.totalEquity.sum() + binance_funcs.get_spot_USDT_balances(client, 'USDT')
# target position
hist = binance_funcs.download_hist(binance_client=client, symbols=['BTCUSDT'],
                                   interval=client.KLINE_INTERVAL_1DAY,
                                   start_str='4 day ago')

data = spark.createDataFrame(hist).select('symbol', 'open_time', 'r')
ts = TechnicalSignal(data, 'open_time', 'symbol')
data = ts.add_max_min_r('r', 0, 1, 'max_r', 'min_r')
data = data.toPandas()
# Order:
binance_funcs.order(client, '1INCH', 20)
