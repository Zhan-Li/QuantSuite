from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
from datetime import date
import importlib
import utilities as utils
importlib.reload(utils)
import os
# This script perform the following tasks
# 1. create trading valume in cash: VOL_C
# 2. create market cap: MKTCAP
# 3. link to stocknames for addtional stock infos such as SHROUT
# 4. link to Compustat GVKEY IID using ccmxpf_lnkhist
# 5. negative prc value means cloing price not avalable, thus using bid-ask average
# Needs pyarrow < 0.15.0 as of 05/05/2020
# groupby does not support DateType(). Need to cast to DatetimeType()
# initiaze pyspark


spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12")

spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()# files to read
dsf = 'data/dsf.sas7bdat'
dse = 'data/dse.sas7bdat'
ccmxpf_lnkhist = 'data/ccmxpf_lnkhist.sas7bdat'
option_file = 'data/opprcd_1996_2020'
# CRSP US stock data
dsf = spark.read.format('com.github.saurfang.sas.spark').load(dsf)
for column in dsf.columns:
    dsf = dsf.withColumnRenamed(column, column.lower())
dsf = dsf \
    .filter(f.col('prc') != 0) \
    .withColumn('permno', f.col('permno').cast(IntegerType())) \
    .withColumn('permco', f.col('permco').cast(IntegerType())) \
    .withColumn('hexcd', f.col('hexcd').cast(IntegerType())) \
    .withColumn('prc', f.abs(f.col('prc'))) \
    .withColumn('vol_d', f.col('prc') * f.col('vol')) \
    .withColumn('shrout', f.col('shrout') * 1000) \
    .withColumn('mktcap', f.col('prc') * f.col('shrout')) \
    .withColumn('ret_o', (1 + f.col('ret')) * f.col('openprc') / f.col('prc') - 1) \
    .withColumn('ret_h', (1 + f.col('ret')) * f.col('askhi') / f.col('prc') - 1) \
    .withColumn('ret_l', (1 + f.col('ret')) * f.col('bidlo') / f.col('prc') - 1)\
    .withColumn('vol_adj', f.col('vol') * f.col('cfacshr')) \
    .withColumn('prc_adj', f.col('prc') / f.col('cfacshr'))
# stocks with options
stock_with_options = spark.read.parquet(option_file)\
    .select('date', 'cusip')\
    .dropDuplicates()\
    .withColumn('optionable', f.lit('Y'))
# get SPY data
SPY = utils.download_return(start_date= '1900-01-01', end_date=date.today(), symbol='SPY')
SPY = spark.createDataFrame(SPY)\
    .select(f.col('Date').alias('date'), f.col('r').alias('ret_SPY'))
# merge SPY and option data
dsf = dsf\
    .join(stock_with_options, on=['date', 'cusip'], how='left')\
    .join(SPY, on='date')
# database to connect permco and permno to ticker and other stock related information.
# note that one multiple PERMNO can correspond to a single one tick with multiple class shares
dse = spark.read.format('com.github.saurfang.sas.spark').load(dse)
for column in dse.columns:
    dse = dse.withColumnRenamed(column, column.lower())
dse = dse.filter(f.col('event') == 'NAMES') \
    .withColumn('namedt', f.col('date')) \
    .withColumn('permno', f.col('permno').cast(IntegerType())) \
    .withColumn('permco', f.col('permco').cast(IntegerType())) \
    .withColumn('shrcd', f.col('shrcd').cast(IntegerType())) \
    .withColumn('permco', f.col('permco').cast(IntegerType())) \
    .withColumn('siccd', f.col('siccd').cast(IntegerType())) \
    .withColumn('naics', f.col('naics').cast(IntegerType())) \
    .select('namedt', 'nameendt', 'permco', 'permno', 'ticker', 'cusip', 'shrcd', 'siccd', 'naics')
dsf = dsf.join(dse, how='left', on=['permno', 'permco', 'cusip']) \
    .filter((f.col('date') >= f.col('namedt')) & (f.col('date') <= f.col('nameendt')))
# data base linking CRSP to compustat
ccm_linktable = spark.read.format('com.github.saurfang.sas.spark').load(ccmxpf_lnkhist)
for column in ccm_linktable.columns:
    ccm_linktable = ccm_linktable.withColumnRenamed(column, column.lower())
ccm_linktable = ccm_linktable.withColumnRenamed('liid', 'iid') \
    .withColumn('permno', f.col('lpermno').cast(IntegerType())) \
    .filter(f.col('permno').isNotNull())\
    .filter((f.col('linkprim') != 'N')) \
    .filter((f.col('linktype') != 'LD') & (f.col('linktype') != 'LX') & (f.col('linktype') != 'LS'))\
    .drop('lpermno') \
    .drop('permco')
dsf = dsf.join(ccm_linktable, how='left', on=['permno'])
dsf = dsf.filter((dsf.date >= dsf.linkdt) & ((dsf.date <= dsf.linkenddt) | (dsf.linkenddt.isNull())))
# export dsf
dsf.withColumn('date', f.col('date').cast(TimestampType())) \
    .filter((f.col('shrcd') == 10) | (f.col('shrcd') == 11) | (f.col('shrcd') == 12) | (f.col('shrcd') == 18) |
            (f.col('shrcd') == 30) | (f.col('shrcd') == 31)) \
    .write.mode('overwrite').parquet('data/dsf_linked')




