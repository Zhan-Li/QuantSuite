from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
import pyspark.sql.functions as f
from pyspark.sql import SparkSession, Window, DataFrame
from data_processor.common import *

usr, pin = get_mysql_secret()
spark = init_spark()
# global params
MILLION = 1000000
# sec file
sec = spark.read.parquet('data/dsf_linked')
# fundq file
fundq = 'data/fundq.sas7bdat'
# CRSP US stock data
fundq = sas_to_parquet(spark, fundq)\
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
    .withColumn('PB', 1/f.col('BM'))\
    .select(*(['date', 'cusip', 'ticker', 'AG', 'BM', 'PB']))
# save data
write_to_mysql(fundq, 'locahost/compustat', 'fundq', usr, pin)




