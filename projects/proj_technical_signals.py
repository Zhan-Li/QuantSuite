from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from quantsuite.signals import TechnicalSignal
from quantsuite import DataManipulator

spark_conf = SparkConf() \
    .set("spark.executor.memory", '50g') \
    .set("spark.driver.memory", '50g') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
# .set("spark.sql.execution.arrow.pyspark.enabled", "true")\
# .set("spark.driver.maxResultSize", "32g") \
# .set("spark.memory.offHeap.size", "16g")
# .set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
# .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12") \
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

dsf = spark.read.parquet('data/dsf_linked').filter(f.col('date') >= '2006-01-01')
ts = TechnicalSignal(dsf, 'date', 'cusip')
dsf_sigs = ts \
    .add_cumr('ret', 1, 1, 'forward_r1d') \
    .add_cumr('ret', 1, 5, 'forward_r1w') \
    .add_cumr('ret', 1, 21, 'forward_r1m') \
    .add_cumr('ret', 1, 126, 'forward_r6m') \
    .add_cumr('ret', 1, 252, 'forward_r12m') \
    .add_cumr('ret', 1, 1, 'forward_r1') \
    .add_cumr('ret', -252, -21, 'r11m') \
    .add_cumr('ret', -147, -21, 'r6m') \
    .add_cumr('ret', -21, 0, 'r1m') \
    .add_52w_high('prc_adj', 'high_52w') \
    .add_std('ret', -21, 0, 'r_std') \
    .add_avg_r('ret', -21, 0, 'r_avg') \
    .add_turnover('vol', 'shrout', 'turnover_daily') \
    .add_illiq('ret', 'vol_d', 'illiq_daily') \
    .add_max_min_r('ret', -21, 0, 'max_r_1m', 'min_r_1m') \
    .add_spread('bid', 'ask', 'spread_daily').data

dsf_sigs = DataManipulator(dsf_sigs, 'date', 'cusip') \
    .add_avg('turnover_daily', -126, 0, 'turnover_6m_avg') \
    .add_rsd('turnover_daily', -126, 0, 'turnover_6m_rsd') \
    .add_avg('vol_d', -126, 0, 'vol_d_6m_avg') \
    .add_rsd('vol_d', -126, 0, 'vol_d_6m_rsd') \
    .add_avg('illiq_daily', -126, 0, 'illiq_6m_avg') \
    .add_skewness('ret', -21, 0, 'skewness_1m') \
    .add_kurtosis('ret', -21, 0, 'kurtosis_1m') \
    .add_avg('spread_daily', -21, 0, 'spread_1m').data

dsf_sigs.write.mode('overwrite').parquet('tech_signals')
