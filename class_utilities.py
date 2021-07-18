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
from pandas.tseries.offsets import MonthEnd

def sas_to_parquet(input, output):
    spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC') \
        .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12")
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    spark.read.format('com.github.saurfang.sas.spark').load(input).write.mode('overwrite').parquet(output)



def read_option(start_year=1996,
                end_year=2020,
                opt_folder='data/option_prices/',
                secnmd_file='data/secnmd.sas7bdat'
                ):
    """
    matches option file with security information.
    Note: Neither symbol or optionid can uniquely identify an option. You need symbol and optionid BOTH to uniquely identify
    """
    ## read option files

    spark_conf = SparkConf() \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC') \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12")
    # .set("spark.sql.execution.arrow.pyspark.enabled", "true")\
    # .set("spark.driver.maxResultSize", "32g") \
    # .set("spark.memory.offHeap.size", "16g")
    # .set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    option = spark.read.format('com.github.saurfang.sas.spark').load(f'{opt_folder}opprcd{start_year}.sas7bdat')
    for option_year in range(start_year + 1, end_year + 1):
        df = spark.read.format('com.github.saurfang.sas.spark').load(f'{opt_folder}opprcd{option_year}.sas7bdat')
        option = option.unionByName(df)

    option = option \
        .withColumn('secid', f.col('secid').cast(IntegerType())) \
        .withColumn('date', f.col('date').cast(TimestampType())) \
        .withColumn('optionid', f.col('optionid').cast(IntegerType())) \
        .withColumn('unique_optionid', f.concat(f.col('optionid'), f.col('symbol')))\
        .withColumn('exdate', f.col('exdate').cast(TimestampType())) \
        .withColumn('exdays', f.datediff(f.col('exdate'), f.col('date'))) \
        .withColumn('strike_price', f.col('strike_price') / 1000) \
        .withColumn('opt_offer_adj', f.col('best_offer') * f.col('cfadj')) \
        .withColumn('opt_bid_adj', f.col('best_bid') * f.col('cfadj')) \
        .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
        .withColumn('opt_spread', (f.col('best_offer') - f.col('best_bid')) / f.col('opt_prc')) \
        .withColumn('opt_vol_adj', f.col('volume') * f.col('contract_size') / f.col('cfadj')) \
        .withColumn('opt_vold', f.col('volume') * f.col('contract_size') * f.col('opt_prc'))\
        .withColumn('opt_prc_adj', f.col('opt_prc') * f.col('cfadj')) \
        .withColumn('opt_prc_adj_lagged', f.lag('opt_prc_adj').over(Window.partitionBy('unique_optionid').orderBy('date'))) \
        .withColumn('opt_r', f.col('opt_prc_adj') / f.col('opt_prc_adj_lagged') - 1) \
        .drop('opt_prc_adj_lagged') \
        .withColumn('open_interest', f.when(f.col('date') <= '2000-11-28', f.col('open_interest'))
                    .otherwise(f.lead('open_interest').over(Window.partitionBy('unique_optionid').orderBy('date')))) \
        .withColumn('open_interest_adj', f.col('open_interest') / f.col('cfadj')) \
        .withColumn('oi_lagged', f.lag(f.col('open_interest_adj')).over(Window.partitionBy('unique_optionid').orderBy('date'))) \
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

    option.write.mode('overwrite').parquet(f'data/opprcd_{start_year}_{end_year}')

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


def get_mom_factors(freq='daily'):
    """
    return a dataframe containing returns for HML, market, SMB
    """
    if freq not in ['daily', 'monthly', 'yearly']:
        raise ValueError('Valid values for freq are daily, monthly, yearly')
    urls = {'daily':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip',
            'monthly':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip',
            'yearly':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'}
    mom_factor = pd.read_csv(urls[freq],  skiprows=13).astype(str).rename(columns = {'Unnamed: 0': 'date' })
    mom_factor.columns = mom_factor.columns.str.strip()
    mom_factor = mom_factor.loc[(mom_factor['Mom'] != '-99.99') & (mom_factor['Mom'] != '-999')]
    time_length = mom_factor['date'].str.strip().str.len()
    if freq == 'monthly':
        mom_factor = mom_factor.loc[time_length == 6]
    elif freq == 'yearly':
        mom_factor = mom_factor.loc[time_length == 4]
    mom_factor = mom_factor.set_index('date')
    mom_factor = mom_factor.dropna()
    return mom_factor




def get_FF_breakpoints(what='ME'):
    """
    return a dataframe containing returns for HML, market, SMB
    """
    urls = {'ME': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/ME_Breakpoints_CSV.zip'}
    break_points = pd.read_csv(urls[what],  skiprows=1, skipfooter=1)
    break_points.columns = ['yearmon', 'n'] + [str(i) + '%' for i in range(5, 105, 5)]
    for col in [i for i in break_points.columns if i not in ['yearmon', 'n']]:
        break_points[col] = break_points[col]*1000000
    return break_points




def get_FF_factors(model = 'FF5', mom = True, freq='daily'):
    """
    return a dataframe containing returns for HML, market, SMB
    """
    if model not in ['FF3', 'FF5', 'FF3-mom', 'FF5-mom']:
        raise ValueError("Valid model: 'FF3', 'FF5', 'FF3-mom', 'FF5-mom'")
    elif ('FF5' in model or mom is True) and freq == 'weekly':
        raise ValueError(f'{model} with mom = True does not have weekly frequency.')
    if freq not in ['daily', 'weekly', 'monthly', 'yearly']:
        raise ValueError('Valid values for freq are daily, weekly, monthly, yearly')

    urls = {
        'FF3':{'daily':
                'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip',
            'weekly':
                'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip',
            'monthly':
                'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip',
            'yearly':
                'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'},
        'FF5': {'daily':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
            'monthly':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip',
            'yearly':
                'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'}
    }

    FF_factors = pd.read_csv(urls[model][freq],  skiprows=3).astype(str).rename(columns = {'Unnamed: 0': 'date' })
    FF_factors = FF_factors.apply(lambda x: x.str.strip())
    FF_factors.columns = FF_factors.columns.str.strip()
    if freq == 'monthly':
        FF_factors = FF_factors.loc[FF_factors['date'].str.len() == 6]
    elif freq == 'yearly':
        FF_factors = FF_factors.loc[FF_factors['date'].str.len() == 4]

    if mom is True:
        mom_factor = get_mom_factors(freq)
        FF_factors = FF_factors.merge(mom_factor, on='date')

    FF_factors = FF_factors.set_index('date')
    FF_factors = FF_factors.astype(float).apply(lambda x: x / 100)
    FF_factors = FF_factors.dropna()
    return FF_factors


def get_q_factors(mom = True, freq='daily'):
    """
    return a dataframe containing returns for HML, market, SMB
    weekly1: weekly calendar
    weekly2: weekly Wed to Wed
    """
    if freq not in ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
        raise ValueError("Valid values for freq are 'daily', 'weekly1', 'weekly2', 'monthly', 'quarterly', 'yearly'")

    urls = {'daily': 'http://global-q.org/uploads/1/2/2/6/122679606/q5_factors_daily_2020.csv',
            'weekly': 'http://global-q.org/uploads/1/2/2/6/122679606/q5_factors_weekly_2020.csv',
            'monthly': 'http://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_2020.csv',
            'quarterly': 'http://global-q.org/uploads/1/2/2/6/122679606/q5_factors_quarterly_2020.csv',
            'yearly': 'http://global-q.org/uploads/1/2/2/6/122679606/q5_factors_annual_2020.csv'

    }

    q_factors = pd.read_csv(urls[freq])
    if freq == 'daily':
        q_factors['date'] = q_factors['DATE']
        q_factors = q_factors.drop(['DATE'], axis=1)
    elif freq =='monthly':
        q_factors['date'] = q_factors['year'].astype(str) + q_factors['month'].astype(str).str.zfill(2)
        q_factors = q_factors.drop(['year', 'month'], axis = 1)
    elif freq == 'quarterly':
        q_factors['date'] = q_factors['year'].astype(str) + q_factors['quarter'].astype(str).str.zfill(2)
        q_factors = q_factors.drop(['year', 'quarter'], axis = 1)
    elif freq == 'yearly':
        q_factors['date'] = q_factors['year']
        q_factors = q_factors.drop(['year'], axis=1)
    q_factors['date'] = q_factors['date'].astype(str)
    q_factors = q_factors.rename(columns={'R_F':'RF', 'R_MKT': 'R_MKT-RF'})

    if mom is True:
        mom_factor = get_mom_factors(freq)
        q_factors = q_factors.merge(mom_factor, on='date')

    q_factors = q_factors.set_index('date')
    q_factors = q_factors.astype(float).apply(lambda x: x / 100)
    q_factors = q_factors.dropna()
    return q_factors



def print_optimization(file):
    """
    print optuna optimization results
    """
    study = joblib.load(file)
    print('Best trial until now:')
    print(' Value: ', study.best_trial.value)
    print(' Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')



def download_return(start_date, end_date, symbols):
    """
    Return Close price and return, Cose price is unadjusted
    """
    if type(symbols) == str:
        symbols = [symbols]
    elif type(symbols) == list:
        symbols = symbols
    hists = pd.DataFrame()
    for symbol in symbols:
        hist = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=False).reset_index()
        hist['r'] = hist[['Adj Close']].pct_change()
        hist['ticker'] = symbol
        hists= hists.append(hist)
    return hists

def allocate_leveraged_asset(r, bechmark_r, leverage: int, weight_target:float, weight_deviation:float,
                             report = True, output='output.html'):
    """
    allocate between a single leveraged asset and fiat
    r: Pandas series returns with time index
    bechmark_r: Pandas series benchmark returns with time index
    fee: percentage trading cost
    """
    weights = [weight_target]
    r = r.dropna()
    weight = weight_target
    levered_r = leverage * r
    for i in levered_r[:-1]:
        weight = weight * (1 + i) / (weight * (1 + i) + 1 - weight)
        weight = weight\
            if np.abs(weight - weight_target) < weight_deviation \
            else weight_target
        weights.append(max(weight, 0))
    result = levered_r.to_frame(name='levered_r')
    result['r'] = r
    result['weight'] = weights
    result['port_r'] = result['weight']*result['levered_r']
    if report == True:
        quantstats.reports.html(result['port_r'], bechmark_r,output=output)
    return result

def get_drawdowns(r):
    """
    r: panda series
    """
    drawdowns = (r + 1).cumprod() / (r + 1).cumprod().cummax() - 1
    return drawdowns.abs()



def get_tradingdays(date: str, start_date = '2019-01-01', end_date='2019-12-31', nth=5, market='NYSE'):
    """
    :param date:
    :param start_year:
    :param end_year:
    :param nth: every nth observation
    :param market:
    :return:
    """
    trading_days = mcal.get_calendar(market).valid_days(start_date, end_date)
    try:
        trading_days_pd = pd.DataFrame({date: list(trading_days)}).iloc[::nth]
        spark= SparkSession.builder.getOrCreate()
        return spark.createDataFrame(trading_days_pd.loc[trading_days_pd[date].notnull()])
    except ValueError as e:
        print(e)

def multi_join(dfs: List[DataFrame], on: List[str], how='inner'):
    """
    consecutive join multiple spark dataframes
    """
    df = dfs[0]
    for i in dfs[1:]:
        df = df.join(i, on=on, how=how)
    return df


def print_duplicates(data: SparkDataframe, cols: List[str]) -> SparkDataframe:
    w = Window.partitionBy(*cols)
    data.select('*', f.count(cols[0]).over(w).alias('dupeCount')) \
        .where('dupeCount > 1') \
        .drop('dupeCount') \
        .show()

def SparktoPandas(data: SparkDataframe):
    data.write.mode('overwrite').parquet('temp')
    df =  pd.read_parquet('temp')
    shutil.rmtree('temp')
    return df

def PandastoSpark(spark, data):
    data.to_parquet('temp.parquet', index=False)
    df = spark.read.parquet('temp.parquet')
    os.remove('temp.parquet')
    return df

def select_CRSP(data):
    return   data.filter((f.col('shrcd') == 10) | (f.col('shrcd') == 11) | (f.col('shrcd') == 12) | (f.col('shrcd') == 18) |
            (f.col('shrcd') == 30) | (f.col('shrcd') == 31))


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

def delete_na_cols(data, na_pct):
    for col in data.columns:
        if sum(data[col].isnull())/len(data) >= na_pct:
            data = data.drop([col], axis = 1)
    return data