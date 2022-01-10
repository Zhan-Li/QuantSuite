# evaluate signal using stocks as investment
# bas performance with 0.61 SR in the 15-20 test period
import pandas as pd
import pyspark.sql.functions as f
from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window

import quantsuite.misc_funcs as utils
from quantsuite import PortfolioAnalysis
from quantsuite.signals import OptionSignal

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
    .set("spark.executor.memory", '50g') \
    .set("spark.driver.memory", "50g") \
    .set("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf) \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/opt_sig.sorting_results") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/opt_sig.sorting_results") \
    .getOrCreate()

# people.write.format("mongo").mode("append").save()


# fixed parameters
train_start = '2010-01-01'
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
    .join(sec, on=['date', 'cusip']) \
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('ks', f.col('strike_price') / f.col('prc')) \
    .withColumn('elasticity', f.col('delta') * f.col('prc') / f.col('opt_prc')) \
    .withColumn('elasticity_abs', f.abs('delta') * f.col('prc') / f.col('opt_prc')) \
    .withColumn('opt_vol_adj+-',
                f.when(f.col('cp_flag') == 'P', -1 * f.col('opt_vol_adj')).otherwise(f.col('opt_vol_adj'))) \
    .withColumn('opt_vold+-', f.when(f.col('cp_flag') == 'P', -1 * f.col('opt_vold')).otherwise(f.col('opt_vold'))) \
    .withColumn('oi_adj_change+-',
                f.when(f.col('cp_flag') == 'P', -1 * f.col('oi_adj_change')).otherwise(f.col('oi_adj_change')))
# option signals

## leverage measures
lvg_sigs = []
for weight in [None, 'opt_vol_adj+-', 'opt_vold+-', 'oi_adj_change+-']:
    lvg_sig1 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', ['C', 'P']) \
        .gen_ks('ks', weight, f'ks|{weight}')
    lvg_sig2 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', ['C', 'P']) \
        .gen_expdays('exdays', weight, f'expdays|{weight}')
    lvg_sig3 = OptionSignal(opt, 'date', 'cusip', 'cp_flag', ['C', 'P']) \
        .gen_elasticity('elasticity', weight, f'elasticity|{weight}')
    lvg_sig = utils.multi_join([lvg_sig1, lvg_sig2, lvg_sig3], on=['date', 'cusip'])
    lvg_sigs.append(lvg_sig)
lvg = utils.multi_join(lvg_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)
## all signals
signals = lvg
# stock return data
sec = spark.read.parquet('data/dsf_linked') \
    .filter(f.col('vol_d') > vold_min) \
    .withColumn(forward_r, f.lead('ret').over(Window.partitionBy('cusip').orderBy('date'))) \
    .select('date', 'cusip', 'ticker', forward_r, 'vol', 'vol_adj', 'vol_d', 'prc', 'ret', 'mktcap', 'shrout',
            'ret_SPY')
# merge stock return with option signal
merged = sec.join(signals, on=['date', 'cusip']).cache()
# Exploration
results = pd.DataFrame()
for sig_col in signals.columns[2:]:
    pa = PortfolioAnalysis(merged, 'date', 'cusip', sig=sig_col, forward_r=forward_r, var_list=vars)
    returns = pa.gen_portr(ntile=10, total_position=0, commission=0, turnover=0)
    sorted = pa.sort_return(days=1, trading_cost=False)
    sorted['sig_name'] = sig_col
    results = results.append(sorted)
    r = returns.loc[returns['sig_rank'] == 'high_minus_low'][forward_r]
results.to_csv('opt_sig_results.csv', mode='a', header=False)
