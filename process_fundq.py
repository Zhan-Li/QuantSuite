from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
from datetime import date
import importlib
import utilities as utils
importlib.reload(utils)
from data_manipulator import DataManipulator
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

spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# global params
MILLION = 1000000
# sec file
sec = spark.read.parquet('data/dsf_linked')
# fundq file
fundq = 'data/fundq.sas7bdat'
# CRSP US stock data
fundq = spark.read.format('com.github.saurfang.sas.spark').option("forceLowercaseNames", True).load(fundq)\
    .withColumn('fyearq', f.col('fyearq').cast(IntegerType()))\
    .withColumn('fqtr', f.col('fqtr').cast(IntegerType()))\
    .withColumn('fyr', f.col('fyr').cast(IntegerType()))\
    .withColumn('cusip', f.col('cusip').substr(0, 8))\
    .withColumn('date', f.greatest('fdateq', 'pdateq', 'rdq'))\
    .filter(f.col('date').isNotNull()) \
    .withColumn('atq_lagged', f.lag('atq').over(Window.partitionBy('cusip').orderBy('date'))) \
    .withColumn('AG', (f.col('atq') - f.col('atq_lagged'))/f.col('atq_lagged'))
# merge with security data
fundq = fundq.join(sec, on=['date', 'cusip'], how='right')
# forward fill fundamental data as it has lower frequency than security data
var_list = ['ceqq', 'AG']
for col in var_list:
    fundq = fundq.withColumn(col, f.last(col, ignorenulls=True).over(Window.partitionBy('cusip').orderBy('date')))
# generate fundamental signals (ratios)
fundq = fundq\
    .withColumn('BM', f.col('ceqq')*MILLION/f.col('mktcap'))\
    .withColumn('PB', 1/f.col('BM'))
# save data
fundq\
    .select(*(['date', 'cusip', 'ticker', 'AG', 'BM', 'PB']))\
    .write.mode('overwrite').parquet('data/fundq')




