# evaluate signal using stocks as investment
import pandas as pd
import pyspark.sql.functions as f
import importlib
from sig_option import OptionMetricsSignal
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortforlioAnalysis
import sig_AMFE; importlib.reload(sig_AMFE)
from sig_AMFE import AMFE
import data_manipulator; importlib.reload(data_manipulator)
import utilities as utils
importlib.reload(utils)
import quantstats
import sig_technical; importlib.reload(sig_technical)
import numpy as np
from sig_technical import TechnicalSignal
from pyspark.sql.types import StringType
import optuna
import joblib
import dask
import dask.dataframe as dd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", '50g') \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
        #.set("spark.sql.execution.arrow.pyspark.enabled", "true")\
        #.set("spark.driver.maxResultSize", "32g") \
        #.set("spark.memory.offHeap.size", "16g")
        #.set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
        #.set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12") \
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# params
ticker = 'SPY'
train_start = '2005-01-01'
train_end = '2013-12-31'
# read option data and stock data
def gen_data(per_contract_fee):
    opt = spark.read.parquet('data/opprcd_1996_2019')\
        .filter(f.col('ticker') == ticker)\
        .withColumn('opt_prc', (f.col('opt_adjoffer') + f.col('opt_adjbid'))/2)
    opt=data_manipulator.prc_to_r(opt, ['optionid'], 'date', 'opt_prc', 'opt_r', per_contract_fee)
    sec = spark.read.parquet('data/dsf_linked')\
        .select('date', 'cusip', 'ret', f.col('prc').alias('stock_prc'))    \
        .withColumn('stock_prc_yesterday', f.lag('stock_prc').over(Window.partitionBy('cusip').orderBy('date'))) \
        .withColumn('stock_r_yesterday', f.lag('ret').over(Window.partitionBy('cusip').orderBy('date')))
    if ticker in ['SPY', 'QQQ']:
        yf_stock = utils.download_return('1996-01-01', '2021-01-31', ticker)
        yf_stock = spark.createDataFrame(yf_stock)\
            .withColumn('ticker', f.lit(ticker))\
            .select(f.col('Date').alias('date'), f.col('r').alias('ret'), f.col('Close').alias('stock_prc'), 'ticker')\
            .withColumn('stock_prc_yesterday', f.lag('stock_prc').over(Window.partitionBy('ticker').orderBy('date'))) \
            .withColumn('stock_r_yesterday', f.lag('ret').over(Window.partitionBy('ticker').orderBy('date')))
        merged = opt.join(yf_stock, on=['date', 'ticker'])
    else:
        merged = opt.join(sec, on=['date', 'cusip'])
    # selecting options based on expiration date and length of the option
    merged = merged \
        .filter(f.col('cp_flag') == 'C') \
        .filter(f.col('contract_size') == 100) \
        .filter(f.col('expiry_indicator').isNull())\
        .withColumn('count', f.count('date').over(Window.partitionBy('optionid')))\
        .filter(f.col('count') >= 28)\
        .withColumn('exp_weekday', f.date_format('exdate', 'E'))\
        .filter((f.col('exp_weekday') == 'Fri')|(f.col('exp_weekday') == 'Sat')) \
        .withColumn('elasticity', f.col('delta') / f.col('opt_prc') * f.col('stock_prc')) \
        .select('date', 'ticker', 'optionid', 'am_settlement', 'expiry_indicator', 'ss_flag', 'contract_size',
                'exdate', 'exdays', 'cp_flag','opt_prc', 'opt_r', 'stock_prc', 'stock_prc_yesterday',
                'stock_r_yesterday', 'ret', 'delta', 'elasticity',  'strike_price')

    train = merged.filter(f.col('date') <= train_end)
    train.write.mode('overwrite').parquet(f'train_fee|{per_contract_fee}.parquet')
    test = merged.filter(f.col('date') > train_end)
    test.write.mode('overwrite').parquet(f'test_fee|{per_contract_fee}.parquet')

gen_data(0)
gen_data(0.75)
# select options to invest
train = spark.read.parquet('train_fee|0.parquet')
# train
def get_port_r(data, e_cutoff, weight_target, weight_deviation, ndays,
               report_logical=False,
               output='ouput.html'):
    # train1, train2 = utils.select_calls(data, 'date', 'ticker', 'optionid', 'exdate', 'exdays', 'strike_price',
    #                                     'stock_prc_yesterday',
    #                                     min_exp_days, min_prc_pct)
    # option = train1 if fixed_strike is False else train2
    option = data \
        .filter(f.col('elasticity') >= e_cutoff) \
        .dropDuplicates(['elasticity']) \
        .withColumn('min_elasticity', f.min('elasticity').over(Window.partitionBy('date'))) \
        .filter(f.col('elasticity') == f.col('min_elasticity'))
    option_sorted = option.toPandas().sort_values('date').set_index('date')
    r = utils.allocate_leveraged_asset(r=option_sorted['opt_r'],
                                       bechmark_r=option_sorted['ret'],
                                       leverage=1,
                                       weight_target=weight_target,
                                       weight_deviation=weight_deviation,
                                       report=report_logical,
                                       output=output)
    port_max_drawdown = utils.get_drawdowns(r['port_r']).max()
    benchmark_max_drawdown = utils.get_drawdowns(option_sorted['ret']).max()
    port_calmar = r['port_r'].sum() / ndays / port_max_drawdown
    benchmark_carlmar = option_sorted['ret'].sum() / ndays / benchmark_max_drawdown
    if port_max_drawdown < benchmark_max_drawdown and \
            port_calmar > benchmark_carlmar and \
            r['port_r'].mean() / r['port_r'].std() > option_sorted['ret'].mean() / option_sorted['ret'].std():
        # avg_r = r['port_r'].sum()/ndays
        avg_r = r['port_r'].mean() / r['port_r'].std() * np.sqrt(252)
        # avg_r = port_calmar
        # avg_r = (r['port_r'] + 1).cumprod().iloc[-1]
    else:
        #avg_r = -port_max_drawdown
        avg_r = r['port_r'].mean() / r['port_r'].std() * np.sqrt(252)
    return avg_r

def objective(trial):
    e_cutoff = trial.suggest_discrete_uniform('e_cutoff', 5, 15, 1)
    weight_target = trial.suggest_uniform('weight_target', 0, 1)
    weight_deviation = trial.suggest_uniform('weight_deviation', 0, 1)
    return get_port_r(train, e_cutoff, weight_target, weight_deviation, ndays,
                       report_logical = False, output = 'output.html')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_jobs = -1)

joblib.dump(study, f"study_elasticity.pkl")

# test best parameters
# Reducing min_prc_pct decrease risk, increases it increase risk, and might significantly reduce returns
# increasing weight target keeps the sharpe ratio, increases risk and return. but increasing too much will reduce return.
# Increasing weight deviation from 0 reduces sharpe ratio.
study_file = 'study_elasticity.pkl'
mystudy = joblib.load(study_file)
best_params = mystudy.best_trial.params

e_cutoff = best_params['e_cutoff']
weight_target = best_params['weight_target']
weight_deviation = best_params['weight_deviation']

e_cutoff = 9
weight_target = 0.2
weight_deviation = 0

get_port_r(train, e_cutoff, weight_target, weight_deviation, ndays, report_logical=True,
           output= f'train|{e_cutoff}|{weight_target}|{weight_deviation}.html')
# test
testdf = spark.read.parquet('test_fee|0.parquet')
option = testdf \
    .filter(f.col('elasticity') >= e_cutoff) \
    .withColumn('min_elasticity', f.min('elasticity').over(Window.partitionBy('date'))) \
    .dropDuplicates(['elasticity']) \
    .filter(f.col('elasticity') == f.col('min_elasticity'))
option_sorted = option.toPandas().sort_values('date').set_index('date')
r = utils.allocate_leveraged_asset(r=option_sorted['opt_r'],
                                   bechmark_r=option_sorted['ret'],
                                   leverage=1,
                                   weight_target=weight_target,
                                   weight_deviation=weight_deviation,
                                   report=True,
                                   output='test.html')


#
stock = utils.download_return('2007-01-01', '2021-01-01', 'QQQ').set_index('Date')
weight_deviation = 0.3
stock_r = utils.allocate_leveraged_asset(r = stock['r'], bechmark_r=stock['r'],
                                   leverage=3, weight_target=1/3, weight_deviation=weight_deviation,
                                   output=f'QQQ{weight_deviation}.html')