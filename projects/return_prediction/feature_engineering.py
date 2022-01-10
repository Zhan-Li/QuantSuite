import json

import pandas as pd
import pyspark.sql.functions as f
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

import quantsuite.misc_funcs as utils
from quantsuite.signals import OptionSignal, TechnicalSignal

# pandas option
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# read SQL credentials
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin = secrets['mysql_password']


# database write function
def write_to_mysql(data, database, table):
    data.write.format('jdbc').options(
        url=f'jdbc:mysql://localhost/{database}',
        driver='com.mysql.cj.jdbc.Driver',
        dbtable=table,
        user=usr,
        password=pin).mode('overwrite').save()


# setup spark
spark_conf = SparkConf() \
    .set("spark.executor.memory", '50g') \
    .set("spark.driver.memory", "50g") \
    .set('spark.driver.maxResultSize', '0') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# prepare data
sec = spark.read.parquet('data/dsf_linked').filter(f.col('date') >= '1996-01-01')

sec = TechnicalSignal(sec, 'date', 'cusip') \
    .add_std('ret', -20, 0, 'ret_std').data
opt = spark.read.parquet('data/opprcd_1996_2020') \
    .filter(f.col('impl_volatility').isNotNull()) \
    .join(sec, on=['date', 'cusip']) \
    .withColumn('ks', f.col('strike_price') / f.col('prc')) \
    .withColumn('p_ratio', f.col('opt_prc') / f.col('prc')) \
    .withColumn('rv-iv', f.col('ret_std') - f.col('impl_volatility')) \
    .withColumn('elasticity', f.col('delta') * f.col('prc') / f.col('opt_prc')) \
    .select('date', 'cusip', 'cp_flag', 'opt_vol_adj', 'opt_vold', 'open_interest_adj', 'oi_adj_change',
            'vol_adj', 'vol_d', 'opt_prc', 'prc', 'p_ratio', 'ks', 'exdays', 'elasticity', 'impl_volatility', 'rv-iv')
# utils.profile_data(opt)


## osv signal
print('OSV' + '-' * 100)
sigs = []
for opt_vol in ['opt_vol_adj', 'opt_vold', 'open_interest_adj', 'oi_adj_change']:
    for sec_vol in ['vol_adj', 'vol_d']:
        opt = opt.withColumn('v_ratio', f.col(opt_vol) / f.col(sec_vol))
        sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=None) \
            .gen_os_v('v_ratio', f'osv|{opt_vol}|{sec_vol}')
        sigs.append(sig)
osv = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(osv, 'osv')
## osp signal
print('osp' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_os_p('p_ratio', f'osp|{weight}')
    sigs.append(sig)
osp = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(osp, 'osp')
## volume signal
print('vol' + '-' * 100)
sigs = []
for vol in ['opt_vol_adj', 'opt_vold', 'open_interest_adj', 'oi_adj_change']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=None) \
        .gen_vol(vol, f'vol|{vol}')
    sigs.append(sig)
vol = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(vol, 'vol')
## ks signal
print('ks' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_ks('ks', f'ks|{weight}')
    sigs.append(sig)
ks = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(ks, 'ks')
## exdays signal
print('exdays' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_expdays('exdays', f'exdays|{weight}')
    sigs.append(sig)
exdays = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(exdays, 'exdays')
## elasticity signal
print('elasticity' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_elasticity('elasticity', f'elasticity|{weight}')
    sigs.append(sig)
elasticity = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(elasticity, 'elasticity')
## IV spread signal
print('IV spread' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_IV('impl_volatility', f'IV|{weight}')
    sigs.append(sig)
IV_spread = utils.multi_join(sigs, on=['date', 'cusip'])
utils.profile_data(IV_spread, file_name='IV_spread.html')
write_to_mysql(IV_spread, 'iv_spread')
## IV skew
print('IV skew' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_IV_skew('impl_volatility', 'ks', f'IV_skew|{weight}')
    sigs.append(sig)
IV_skew = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(IV_skew, 'iv_skew')
## rv-iv
print('rv-iv' + '-' * 100)
sigs = []
for weight in [None, 'opt_vol_adj', 'opt_vold', 'open_interest_adj']:
    sig = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight=weight) \
        .gen_rv_iv_spread('rv-iv', f'rv-iv|{weight}')
    sigs.append(sig)
RV_IV = utils.multi_join(sigs, on=['date', 'cusip'])
write_to_mysql(RV_IV, 'rv_iv')
