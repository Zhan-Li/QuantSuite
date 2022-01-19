import re

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from binance.client import Client
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window

import quantsuite.trader.binance as binance_funcs
from quantsuite import DataManipulator
from quantsuite.signals import TechnicalSignal

# pandas options
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# params
numOfAssets = 15
percentInvest = 0.9
# spark
spark_conf = SparkConf() \
    .set("spark.executor.memory", '30g') \
    .set("spark.driver.memory", "30g") \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# binance API
api_key = 'api_ky'
api_secret = 'api_secret'
client = Client(api_key, api_secret)
sigs = ['avg_sigComov|-10|0_down_20', 'sigcumr|0|0']

# current position
isolated_margin_account = binance_funcs.get_isolated_margin_account(client, 'USDT')
currentPosition = isolated_margin_account[['symbol', 'base_netAssetOfQuote']]
totalMoney = isolated_margin_account.totalEquity.sum() + binance_funcs.get_spot_USDT_balances(client, 'USDT')
# isolated_margin_account[(isolated_margin_account['base_totalAsset']!=0) | (isolated_margin_account['quote_totalAsset']!=0)]
# generate signals
isolated_margin_USDT = binance_funcs.get_all_isolated_margin_symbols(client)
isolated_margin_USDT = [i + 'USDT' for i in isolated_margin_USDT]
hist = binance_funcs.download_hist(binance_client=client, symbols=isolated_margin_USDT,
                                   interval=client.KLINE_INTERVAL_6HOUR,
                                   start_str='200 hours ago')
data = spark.createDataFrame(hist).select('symbol', 'open_time', 'r')
BTC = data.filter(f.col('symbol') == 'BTCUSDT').select('open_time', f.col('r').alias('r_BTC'))
data = data.join(BTC, on='open_time')

data = DataManipulator().add_r(data, 'symbol', 'open_time', 'r', 0, 0, 'lastr')
data = DataManipulator().add_cross_section_zscore(data, 'open_time', 'lastr', 'lastr_zscore')
data = TechnicalSignal(data, 'symbol', 'open_time', 'r', -10, 0) \
    .add_comovement('r_BTC', 'comov').drop('comov', 'comov_up')
data = DataManipulator().add_avg(data, 'symbol', 'open_time', 'comov_down', -20, 0, 'comov_down_avg')
data = DataManipulator().add_cross_section_zscore(data, 'open_time', 'comov_down_avg', 'comov_down_avg_zscore')
data = data.withColumn('sig_merged', (-f.col('lastr_zscore') - f.col('comov_down_avg_zscore')) / 2)
data = data \
    .withColumn('max_open_time', f.max('open_time').over(Window.partitionBy('symbol'))) \
    .filter(f.col('open_time') == f.col('max_open_time')) \
    .select('open_time', 'symbol', 'sig_merged') \
    .toPandas()
# generate position
merged = data.merge(currentPosition, on='symbol', how='inner')
merged['sig_rank'] = merged['sig_merged'].rank(method='first')
conditions = [merged['sig_rank'] <= numOfAssets, merged['sig_rank'] > len(merged) - numOfAssets]
choices = [-np.round(totalMoney * percentInvest / numOfAssets, 2),
           np.round(totalMoney * percentInvest / numOfAssets, 2)]
merged['targetPosition'] = np.select(conditions, choices, default=0)
merged.to_csv('position.csv')
# Order execution:
firstOrders = merged.loc[merged['targetPosition'] == 0]
for index, row in firstOrders.iterrows():
    base_symbol = re.search('(.*)USDT', row['symbol']).group(1)
    target_position = row['targetPosition']
    binance_funcs.order(client, base_symbol, target_position)

secondOrders = merged.loc[merged['targetPosition'] < 0]
for index, row in secondOrders.iterrows():
    base_symbol = re.search('(.*)USDT', row['symbol']).group(1)
    print('Processing', base_symbol)
    target_position = row['targetPosition']
    binance_funcs.order(client, base_symbol, target_position)

thirdOrders = merged.loc[merged['targetPosition'] > 0]
for index, row in thirdOrders.iterrows():
    base_symbol = re.search('(.*)USDT', row['symbol']).group(1)
    print('Processing', base_symbol)
    target_position = row['targetPosition']
    binance_funcs.order(client, base_symbol, target_position)

binance_funcs.order(client, base_symbol, target_position)

# helper function to reduce all position to 0
# for index, row in merged.iterrows():
#     base_symbol = re.search('(.*)USDT', row['symbol']).group(1)
#     print('Processing', base_symbol)
#     binance_funcs.order(client, base_symbol, 0,filters)

from binance import BinanceSocketManager

bm = BinanceSocketManager(client)


# start any sockets here, i.e a trade socket
def process_message(msg):
    print("message type: {}".format(msg['e']))
    print(msg)
    # do something


conn_key = bm.start_isolated_margin_socket('BTCUSDT', process_message)
# then start the socket manager
bm.start()
