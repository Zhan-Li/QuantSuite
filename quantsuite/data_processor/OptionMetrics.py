from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
import pyspark.sql.functions as f
from data_processor.common import *


usr, pin = get_mysql_secret()
spark = init_spark()
#params
start_year=1996
end_year=2020
opt_folder='data/option_prices/'
secnmd_file='data/secnmd.sas7bdat'
CHECK_DATA = False
# combine all option files

option = sas_to_parquet(spark, f'{opt_folder}opprcd{start_year}.sas7bdat')
for option_year in range(start_year + 1, end_year + 1):
    df = sas_to_parquet(spark,f'{opt_folder}opprcd{option_year}.sas7bdat')
    option = option.unionByName(df)
# process option data
option = option \
    .withColumn('contract_size', f.when(f.col('contract_size') < 0, None).otherwise(f.col('contract_size'))) \
    .withColumn('forward_price', f.when(f.col('forward_price') < 0, None).otherwise(f.col('forward_price'))) \
    .withColumn('cp_flag', f.when(f.col('cp_flag') == 'c', 'C').otherwise(f.col('cp_flag'))) \
    .withColumn('secid', f.col('secid').cast(IntegerType())) \
    .withColumn('date', f.col('date').cast(TimestampType())) \
    .withColumn('optionid', f.col('optionid').cast(IntegerType())) \
    .withColumn('unique_optionid', f.concat(f.col('optionid'), f.col('symbol'))) \
    .withColumn('exdate', f.col('exdate').cast(TimestampType())) \
    .withColumn('exdays', f.datediff(f.col('exdate'), f.col('date'))) \
    .withColumn('strike_price', f.col('strike_price') / 1000) \
    .withColumn('opt_offer_adj', f.col('best_offer') * f.col('cfadj')) \
    .withColumn('opt_bid_adj', f.col('best_bid') * f.col('cfadj')) \
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('opt_spread', (f.col('best_offer') - f.col('best_bid')) / f.col('opt_prc')) \
    .withColumn('opt_spread', f.when(f.col('opt_spread') < 0, None).otherwise(f.col('opt_spread'))) \
    .withColumn('opt_vol_adj', f.col('volume') * f.col('contract_size') / f.col('cfadj')) \
    .withColumn('opt_vold', f.col('volume') * f.col('contract_size') * f.col('opt_prc')) \
    .withColumn('opt_prc_adj', f.col('opt_prc') * f.col('cfadj')) \
    .withColumn('opt_prc_adj_lagged', f.lag('opt_prc_adj').over(Window.partitionBy('unique_optionid').orderBy('date'))) \
    .withColumn('opt_r', f.col('opt_prc_adj') / f.col('opt_prc_adj_lagged') - 1) \
    .drop('opt_prc_adj_lagged') \
    .withColumn('open_interest', f.when(f.col('date') <= '2000-11-28', f.col('open_interest'))
                .otherwise(f.lead('open_interest').over(Window.partitionBy('unique_optionid').orderBy('date')))) \
    .withColumn('open_interest_adj', f.col('open_interest') / f.col('cfadj')) \
    .withColumn('oi_lagged',
                f.lag(f.col('open_interest_adj')).over(Window.partitionBy('unique_optionid').orderBy('date'))) \
    .withColumn('oi_adj_change', f.col('open_interest_adj') - f.col('oi_lagged')) \
    .drop('oi_lagged')

secnmd = spark.read.format('com.github.saurfang.sas.spark').load(secnmd_file) \
    .drop('class', 'issuer', 'issue') \
    .withColumn('secid', f.col('secid').cast(IntegerType())) \
    .withColumn('effect_date', f.col('effect_date').cast(TimestampType()))

option = option \
    .join(secnmd, on='secid', how='inner') \
    .filter(f.col('date') >= f.col('effect_date')) \
    .withColumn('most_recent_effect_date', f.max('effect_date').over(Window.partitionBy('date', 'secid'))) \
    .filter(f.col('effect_date') == f.col('most_recent_effect_date')) \
    .drop('effect_date', 'most_recent_effect_date')

write_to_mysql(option, 'locahost/optionmetrics', 'optionmetrics', usr, pin)
