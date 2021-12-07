import pyspark.sql.functions as f
from pyspark import StorageLevel
import importlib
import class_sig_option
importlib.reload(class_sig_option)
from class_sig_option import OptionSignal
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import class_sig_evaluator; importlib.reload(class_sig_evaluator)
import funcs_utilities as utils
import class_sig_technical; importlib.reload(class_sig_technical)
import class_data_manipulator
importlib.reload(class_data_manipulator)
from class_data_manipulator import DataManipulator
import matplotlib.pyplot as plt
import datetime

# setup spark
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

# prepare stock and option data
sec = spark.read.parquet('data/dsf_linked')\
    .select('date', 'cusip', 'vol_adj', 'vol_d', 'prc')
opt = spark.read.parquet('data/opprcd_1996_2020') \
    .join(sec, on=['date', 'cusip'])\
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('ks', f.col('strike_price')/f.col('prc')) \
    .withColumn('elasticity', f.abs('delta') * f.col('prc') / f.col('opt_prc'))
# generate option signals
## os signal
os_sigs = []
for opt_vol in ['opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    for sec_vol in ['vol_adj', 'vol_d']:
        opt = opt.withColumn('os', f.col(opt_vol) / f.col(sec_vol))
        os_sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P') \
            .gen_os_v('os', f'os|{opt_vol}|{sec_vol}')
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