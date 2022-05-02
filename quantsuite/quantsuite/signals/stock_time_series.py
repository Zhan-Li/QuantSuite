import pyspark.sql.functions as f
from pyspark.sql import DataFrame as SparkDataFrame


class TimeSeriesTechnicalSignal:
    """
    The OptionSignal class gnerates option-based trading signals.
    """

    def __init__(self, data: SparkDataFrame, time: str, name: str):
        self.data, self.name, self.time = data, name, time

    def gen_mkt_breadth(self, r, mkt_breath: str):
        """
        start, end as the star, end in rowsBetween, both start and end are inclusive.
        """
        return self.data \
            .withColumn('ret_positive', f.when(f.col(r) >= 0, 1).otherwise(0)) \
            .groupby(self.time) \
            .agg((f.sum('ret_positive') / f.count('ret_positive')).alias(mkt_breath))
