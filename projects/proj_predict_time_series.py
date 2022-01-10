# evaluate signal using stocks as investment

import matplotlib.pyplot as plt
import pandas as pd
import pyfolio
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, col, when
from quantsuite.signals import OptionSignal
from quantsuite import DataManipulator
import quantsuite.misc_funcs as utils
from quantsuite import PerformanceEvaluation

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# init spark
spark_conf = SparkConf() \
    .set("spark.driver.memory", "28g") \
    .set("spark.executor.memory", "8g") \
    .set('spark.driver.maxResultSize', '4g') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC') \
    .set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# parameters
start_year = 2007
end_year = 2010
cp_flag = 'C'
sig_name = 'elasticity'
freq = 'M'
exdays_max = float('inf')
exdays_min = 0
opt_vol_min = 0
opt_prc_min = 0
# read stock returns
sec = spark.read.parquet('/Volumes/Data/Google Drive/WRDS/CRSP/dsf_linked').filter(year('date') >= 2007) \
    .select('date', 'cusip', 'ret', 'ticker', 'prc', 'vol_avg', 'vol_c_avg', 'stddev', 'mktcap') \
    .filter(col('ticker') == 'SPY')
# generate signal
opt = spark.read.parquet(f'/Volumes/Data/Google Drive/WRDS/OPTIONM/opprcd_2007_2010') \
    .unionByName(spark.read.parquet(f'/Volumes/Data/Google Drive/WRDS/OPTIONM/opprcd_2011_2014')) \
    .filter(year('date') == 2007)
sig = OptionSignal(opt_file=opt, sec_file=sec) \
    .gen_elasticity_sig(sig_col=sig_name, delta_col='delta', sec_prc='prc', vol_above='opt_vol', vol_below='opt_vol',
                        exdays_max=exdays_max, exdays_min=exdays_min, opt_vol_min=opt_vol_min,
                        opt_prc_min=opt_prc_min).cache()
sig_avg = DataManipulator().gen_avg(data=sig, name=['cusip', 'cp_flag'], time='date', value='elasticity', lookback=4,
                                   avg_col='elasticity_avg')
sig_C = sig_avg.filter(col('cp_flag') == 'C') \
    .withColumnRenamed('elasticity', 'elasticity_C') \
    .withColumnRenamed('elasticity_avg', 'elasticity_avg_C')
sig_P = sig_avg.filter(col('cp_flag') == 'P') \
    .withColumnRenamed('elasticity', 'elasticity_P') \
    .withColumnRenamed('elasticity_avg', 'elasticity_avg_P')
sig_merged = sig_P.join(sig_C, on=['date', 'cusip'], how='inner') \
    .withColumn('elasticity_spread1', (col('elasticity_C') + col('elasticity_P'))) \
    .withColumn('elasticity_spread2',
                (col('elasticity_C') + col('elasticity_P')) / (col('elasticity_C') - col('elasticity_P')))
# general return
sec_cumr =  DataManipulator().gen_cum_r(data=sec, name=['cusip'], value='ret', time='date', cum_r='cum_r')
sec_r =  DataManipulator().gen_forward_r(data=sec_cumr, name=['cusip'], time='date', bid='cum_r', offer='cum_r').cache()
## merge signals with stocks
merged = sig_merged \
    .join(sec_r, on=['cusip', 'date'], how='inner').sort('cusip', 'date').cache()

# trade
quantile_C = merged.approxQuantile("elasticity_C", [0.5], 0)[0]
quantile_P = merged.approxQuantile("elasticity_P", [0.5], 0)[0]
merged = merged \
    .withColumn('position',
                when((col('elasticity_C') >= quantile_C) | (col('elasticity_P') <= quantile_P), 1).otherwise(0)) \
    .withColumn('r', col('position') * col('forward_r'))
df = merged.toPandas().set_index('date')
# evaluation
r = 'r'
pyfolio.create_returns_tear_sheet(df[r])
plt.savefig('yo.png')
eva = PerformanceEvaluation(df[[r]], r, 'daily')
eva.get_full_perf()
# compared with the market
benchmarket = utils.download_return(2007, 2014, 'SPY')
eva = PerformanceEvaluation(benchmarket, 'r', 'daily')
eva.get_full_perf()
