# this script provnamees various stats function used on spark dataframes
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
from pyspark.conf import SparkConf
from pyspark.sql import functions as f
import joblib
import yfinance as yf
from scipy import stats
import sys
import empyrical
from pandas_profiling import ProfileReport
import pandas_market_calendars as mcal
from typing import List
import utilities as utils


class DataManipulator:
    def __init__(self, data: SparkDataFrame, time: str, name: str):
        self.data, self.time, self.name = data, time, name

    def w(self, start, end):
        return Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end)

    def filter(self, df, start, end):
        """
        In rolling calculation, Pyspark will generate a value even the total number of obsevations is smaller than the
        window size in the beginning of the period.
        """
        return df\
            .withColumn('count', f.count(self.time).over(self.w(start, end)))\
            .filter(f.col('count') >= end-start+1)\
            .drop('count')

    def add_timeseries_comparison(self, value: str, start: int, end: int, zscore='zscore', ratio='ratio'):
        """
        generate time_series z-score and value to average ratio
        """
        self.data = self.data \
            .withColumn('avg', f.mean(value).over(self.w(start, end))) \
            .withColumn('std', f.stddev(value).over(self.w(start, end))) \
            .withColumn(zscore, (f.col(value) - f.col('avg')) / f.col('std')) \
            .withColumn(ratio, f.col(value)/f.col('avg'))\
            .drop('avg', 'std')
        self.data = self.filter(self.data, start, end)
        return self


    def add_cross_section_zscore(self, value: str, zscore:str):
        """
        generate time_series z-score
        return a df with a new column which is a zscore of another column
        """
        self.data = self.data \
            .withColumn('avg', f.mean(value).over(Window.partitionBy(self.time))) \
            .withColumn('std', f.stddev(value).over(Window.partitionBy(self.time))) \
            .withColumn(zscore, (f.col(value) - f.col('avg')) / f.col('std')) \
            .drop('avg', 'std')
        return self

    def resample(self, nth: int):
        """
        nth: resample very nth row
        """
        self.data = self.data \
            .withColumn('index', f.row_number().over(Window.partitionBy(self.name).orderBy(self.time))) \
            .filter(f.col('index') % nth == 0)\
            .drop('index')
        return self

    def add_cumr(self, r: str, start: int, end: int, cum_r: str):
        """
        return a df with a new column which is a cumulative product of another column
        """
        self.data = self.data \
            .withColumn(cum_r, f.sum(f.log(f.col(r) + 1)).over(self.w(start, end))) \
            .withColumn(cum_r, f.exp(f.col(cum_r))-1)
        self.data = self.filter(self.data, start, end)
        return self

    def add_sum(self, value: str, start: int, end: int, sum_col: str):
        """
        end of period wealth assuming starting wealth of 1
        """
        self.data = self.data.withColumn(sum_col, f.sum(value).over(self.w(start, end)))
        self.data = self.filter(self.data, start, end)
        return self

    def add_avg(self, value: str, start: int, end: int, avg_col='avg'):
        self.data = self.data.withColumn(avg_col, f.mean(value).over(self.w(start, end)))
        self.data = self.filter(self.data, start, end)
        return self

    def add_diff(self, value: str, diff: str, lag=1):
        self.data = self.data \
            .withColumn('value_lagged', f.lag(f.col(value), offset=lag).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn(diff, f.col(value) - f.col('value_lagged')) \
            .drop('value_lagged')
        return self

    def add_wealth(self, r: str, start: int, end: int, wealth: str):
        """
        end of period wealth assuming starting wealth of 1
        """
        self.data = self.data \
            .withColumn(wealth, f.sum(f.log(f.col(r) + 1)).over(self.w(start, end))) \
            .withColumn(wealth, f.exp(f.col(wealth)))
        self.data = self.filter(self.data, start, end)
        return self


    def prc_to_r(self, price: str, r: str, fee: float, lag=1):
        """
        fee: actual fee, not percentage
        """
        self.data = self.data \
            .withColumn('value_lagged', f.lag(f.col(price), offset=lag).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn(r, (f.col(price)*100 - f.col('value_lagged')*100 - 2*fee)/(f.col('value_lagged')*100 )) \
            .drop('value_lagged')
        return self

