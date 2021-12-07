from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
import pyspark.sql.functions as f
from data_processor.common import *

usr, pin = get_mysql_secret()
spark = init_spark()
# process CRSP mutual fund dataset
mf_file = 'data/daily_nav_ret.sas7bdat'
fund_summary_file = 'data/fund_summary2.sas7bdat'
mf = sas_to_parquet(spark, mf_file) \
    .withColumn('crsp_fundno', f.col('crsp_fundno').cast(IntegerType()))
fund_summary =  sas_to_parquet(spark, fund_summary_file)\
    .select(f.col('crsp_fundno').cast(IntegerType()),'ticker', 'et_flag')\
    .filter(f.col('ticker').isNotNull())\
    .dropDuplicates()
merged = mf.join(fund_summary, on='crsp_fundno')
write_to_mysql(merged, 'locahost/crsp', 'mutual_fund', usr, pin)
