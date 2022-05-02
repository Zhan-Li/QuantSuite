import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType, TimestampType

from quantsuite.data_processor.common import *

usr, pin = get_mysql_secret()
spark = init_spark()
# process CRSP stock dataset
dsf_file = 'data/dsf.sas7bdat'
dse_file = 'data/dse.sas7bdat'
ccmxpf_lnkhist = 'data/ccmxpf_lnkhist.sas7bdat'
option_file = 'data/opprcd_1996_2020'
# CRSP US stock data
dsf = sas_to_parquet() \
    .filter(f.col('prc') != 0) \
    .withColumn('permno', f.col('permno').cast(IntegerType())) \
    .withColumn('permco', f.col('permco').cast(IntegerType())) \
    .withColumn('hexcd', f.col('hexcd').cast(IntegerType())) \
    .withColumn('date', f.col('date').cast(TimestampType())) \
    .filter(f.col('date') >= '1970-01-01') \
    .withColumn('prc', f.abs(f.col('prc'))) \
    .withColumn('vol_d', f.col('prc') * f.col('vol')) \
    .withColumn('shrout', f.col('shrout') * 1000) \
    .withColumn('mktcap', f.col('prc') * f.col('shrout')) \
    .withColumn('ret_o', (1 + f.col('ret')) * f.col('openprc') / f.col('prc') - 1) \
    .withColumn('ret_h', (1 + f.col('ret')) * f.col('askhi') / f.col('prc') - 1) \
    .withColumn('ret_l', (1 + f.col('ret')) * f.col('bidlo') / f.col('prc') - 1) \
    .withColumn('vol_adj', f.col('vol') * f.col('cfacshr')) \
    .withColumn('prc_adj', f.col('prc') / f.col('cfacshr'))
# stocks with options
stock_with_options = spark.read.parquet(option_file) \
    .select('date', 'cusip') \
    .dropDuplicates() \
    .withColumn('optionable', f.lit('Y'))
# merge SPY and option data
dsf = dsf \
    .join(stock_with_options, on=['date', 'cusip'], how='left')
# database to connect permco and permno to ticker and other stock related information.
# note that one multiple PERMNO can correspond to a single one tick with multiple class shares
dse = sas_to_parquet(spark, dse_file) \
    .filter(f.col('event') == 'NAMES') \
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
ccm_linktable = sas_to_parquet(spark, ccmxpf_lnkhist) \
    .withColumnRenamed('liid', 'iid') \
    .withColumn('permno', f.col('lpermno').cast(IntegerType())) \
    .filter(f.col('permno').isNotNull()) \
    .filter((f.col('linkprim') != 'N')) \
    .filter((f.col('linktype') != 'LD') & (f.col('linktype') != 'LX') & (f.col('linktype') != 'LS')) \
    .drop('lpermno') \
    .drop('permco')
dsf = dsf.join(ccm_linktable, how='left', on=['permno'])
dsf = dsf.filter((dsf.date >= dsf.linkdt) & ((dsf.date <= dsf.linkenddt) | (dsf.linkenddt.isNull())))
# export dsf
dsf = dsf.withColumn('date', f.col('date').cast(TimestampType())) \
    .filter((f.col('shrcd') == 10) | (f.col('shrcd') == 11) | (f.col('shrcd') == 12) | (f.col('shrcd') == 18) |
            (f.col('shrcd') == 30) | (f.col('shrcd') == 31))

write_to_mysql(dsf, 'localhost/crsp', 'stock', usr, pin)
