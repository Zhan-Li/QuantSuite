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
mf = 'data/daily_nav_ret.sas7bdat'
fund_summary = 'data/fund_summary2.sas7bdat'
# CRSP US stock data
mf = spark.read.format('com.github.saurfang.sas.spark').option("forceLowercaseNames", True).load(mf)\
    .withColumn('crsp_fundno', f.col('crsp_fundno').cast(IntegerType()))
fund_summary = spark.read.format('com.github.saurfang.sas.spark').option("forceLowercaseNames", True)\
    .load(fund_summary)\
    .select(f.col('crsp_fundno').cast(IntegerType()),'ticker', 'et_flag')\
    .filter(f.col('ticker').isNotNull())\
    .dropDuplicates()
# merge

merged = mf.join(fund_summary, on='crsp_fundno')
merged.write.mode('overwrite').parquet('mf.parquet')