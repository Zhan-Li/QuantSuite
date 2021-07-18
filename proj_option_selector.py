import pandas as pd
from pandas import DataFrame as pdDataframe
from pyspark.sql import DataFrame as SparkDataframe
import numpy as np
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType
from pyspark.conf import SparkConf
import pyspark.sql.functions as f
import joblib
import yfinance as yf
from typing import List
import os
import pandas_market_calendars as mcal
import psutil
import shutil
import quantstats


class OptionSelector:
    def __init__(self, data, time, sec_name, cp_flag):
        self.data = data
        self.time = time
        self.sec_name = sec_name
        self.cp_flag = cp_flag

    def by_max(self, value1: str, value2: str):
        window = Window.partitionBy(self.time, self.sec_name, self.cp_flag)
        return self.data\
            .withColumn('max_value', f.max(value1).over(window))\
            .filter(f.col(value1) == f.col('max_value')) \
            .withColumn('max_value', f.max(value2).over(window)) \
            .filter(f.col(value2) == f.col('max_value'))\
            .withColumn('row', f.row_number()
                        .over(Window.partitionBy(self.time, self.sec_name, self.cp_flag).orderBy(value1)))\
            .filter(f.col('row')==1)\
            .drop('max_value', 'row')

    def select_option(df, time_col, id_col, cp_flag_col, vol_col, strick_prc_col, exdays_col):
        """
        select option for investment. There is only one call and put selected to invest each day
        """
        return df.withColumn('max_vol', f.max(vol_col)
                             .over(Window.partitionBy([time_col, id_col, cp_flag_col]))) \
            .filter(f.col(vol_col) == f.col('max_vol')) \
            .withColumn('max_strike', f.max(strick_prc_col)
                        .over(Window.partitionBy([time_col, id_col, cp_flag_col]))) \
            .filter(f.col(strick_prc_col) == f.col('max_strike')) \
            .withColumn('min_exdays', f.max(exdays_col)
                        .over(Window.partitionBy([time_col, id_col, cp_flag_col]))) \
            .filter(f.col(exdays_col) == f.col('min_exdays'))



    def select_calls(data, time:str, stock_id:str, option_id:str,
                    exp_date: str, exp_days: str, strik_price: str, stock_prc_lag:str,
                    min_exp_days: int, min_strike_delta: float) -> tuple:

        window = Window.partitionBy(time, stock_id)
        select_exdays = data \
            .filter(f.col(exp_days) >= min_exp_days) \
            .withColumn('min_exp_days', f.min(exp_days).over(window)) \
            .filter(f.col(exp_days) == f.col('min_exp_days')) \
            .drop('min_exp_days')
        select_strike_variable = select_exdays\
            .filter(f.col(strik_price) >= f.col(stock_prc_lag) * (1 + min_strike_delta)) \
            .withColumn('min_call_strike', f.min(strik_price).over(window)) \
            .filter(f.col(strik_price) == f.col('min_call_strike')) \
            .drop('min_call_strike')
        calls_fixed = select_strike_variable\
            .withColumn(option_id, f.first(option_id).over(Window.partitionBy(exp_date).orderBy(time))) \
            .select(time, option_id)
        select_strike_fixed = calls_fixed.join(data, on=[time, option_id])
        return select_strike_variable, select_strike_fixed

    def select_puts(data, time:str, stock_identifier:str, min_exp_days: int, min_strike_pct: float,
                    exp_days: str, strik_price: str, stock_prc: str) -> SparkDataframe:
        window = Window.partitionBy(time, stock_identifier)
        select_exdays = data \
            .filter(f.col(exp_days) >= min_exp_days) \
            .withColumn('min_exp_days', f.min(exp_days).over(window)) \
            .filter(f.col(exp_days) == f.col('min_exp_days')) \
            .drop('min_exp_days')
        select_strike = select_exdays\
            .withColumn('max_put_strike', f.col(stock_prc) * (1 - min_strike_pct)) \
            .filter(f.col(strik_price) <= f.col('max_put_strike')) \
            .withColumn('max_put_strike', f.max(strik_price).over(window)) \
            .filter(f.col(strik_price) == f.col('max_put_strike'))\
            .drop('max_put_strike')
        return select_strike