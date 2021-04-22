# evaluate signal using stocks as investment
# bas performance with 0.61 SR in the 15-20 test period
import pandas as pd
import pyspark.sql.functions as f
import importlib
from sig_option import OptionSignal
import sig_option
importlib.reload(sig_option)
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortfolioAnalysis
import utilities as utils
import quantstats
import sig_technical; importlib.reload(sig_technical)
import numpy as np
from sig_technical import TechnicalSignal
import optuna
import joblib
import data_manipulator
importlib.reload(data_manipulator)
from data_manipulator import DataManipulator
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
        #.set("spark.sql.execution.arrow.pyspark.enabled", "true")\
        #.set("spark.driver.maxResultSize", "32g") \
        #.set("spark.memory.offHeap.size", "16g")
        #.set('spark.sql.execution.arrow.maxRecordsPerBatch', 50000)
        #.set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12") \
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# fixed parameters
train_start = '2005-01-01'
train_end = '2015-12-31'
forward_r = 'forward_r'
vol_avg = 'vol_avg'
ret_avg = 'ret_avg'
vold_min = 200000
total_position = 60000
# read stock data
sec = spark.read.parquet('data/dsf_linked')\
    .withColumn('ret_avg', (f.abs('ret') + f.abs('ret_o') + f.abs('ret_h') + f.abs('ret_l')) / 4)\
    .filter(f.col('date') >= train_start)\
    .filter(f.col('vol_d') >= vold_min)\
    .select('date', 'cusip', 'ticker', 'vol', 'vol_adj', 'prc', 'ret', 'ret_avg', 'ret_SPY')


AAPL=sec.filter(f.col('ticker') == 'AAPL').filter(f.col('date') >= '2020-12-01')
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['ret_avg'], outputCol = 'features').setHandleInvalid("skip")
vhouse_df = vectorAssembler.transform(AAPL)
lr = LinearRegression(featuresCol='features', labelCol='vol_adj')
lr_model = lr.fit(vhouse_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)




# code requires change -------------------------------
# variable params
sig_name = 'v_ratio'
output_file = 'v_ratio.pkl'
def objective(crsp_data, start, end, max_ret, n, days, sig_rank_selected='high_minus_low', train = True,
              report=False, filename='report.html', low_minus_high=False, trading_cost=True, absolute_SR=True):
    crsp_data = DataManipulator(crsp_data, 'date', 'cusip') \
        .add_cumr('ret', 1, days, forward_r) \
        .add_avg('vol_adj', -(days - 1), 0, vol_avg) \
        .add_avg('ret_avg', -(days - 1), 0, ret_avg) \
        .resample(days) \
        .data
    train_data = crsp_data.filter(f.col('date') <= train_end)
    test_data = crsp_data.filter(f.col('date') > train_end)
    data = train_data if train is True else test_data

    data = TechnicalSignal(data, 'date', 'cusip')\
        .add_value_average_ratio('vol_adj', start, end, sig_name)\
        .data\
        .filter(f.col('ret_avg') <= max_ret)
    pa = PortfolioAnalysis(data, 'date', 'cusip', sig=sig_name, forward_r=forward_r)
    returns = pa.gen_univariate_portr2(n, total_position, 1, 1)

    r_selected = returns.loc[returns['sig_rank'] == sig_rank_selected]
    r_cost = r_selected['trading_cost'][::days] if trading_cost is True else 0
    r_raw = r_selected[forward_r][::days]
    r = r_raw - r_cost
    sr = r.mean()/r.std()*np.sqrt(252/days)
    sr = np.abs(sr) if absolute_SR is True else sr
    if report is True:
        if low_minus_high is True:
            r_selected = -r_selected
        quantstats.reports.html(r, 'SPY', output=filename)
        print(pa.sort_return(days=days, trading_cost=trading_cost))
        return returns
    else:
        if sr == np.nan:
            return 0
        if sr != np.nan:
            return sr

def optimize(trial):
    start = trial.suggest_int('start', -365, 0, 10)
    end = trial.suggest_int('end', start, 0, 1)
    max_ret = trial.suggest_discrete_uniform('max_ret', 0.01, 0.1, 0.001)
    n = trial.suggest_int('n', 1, 100, 10)
    days = trial.suggest_int('days', 1, 5, 1)
    return objective(sec, start, end,  max_ret, n,  days, sig_rank_selected='high', absolute_SR=False)

# search for parameter -------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(optimize,  n_jobs=-1, n_trials=10000)
study_df = study.trials_dataframe()
joblib.dump(study, output_file)

# read optimized result
study_file = output_file
mystudy = joblib.load(study_file)
best_params = mystudy.best_trial.params
print(best_params)
best_params_adjusted = {'start': -75, 'end': -1, 'max_ret': 0.03, 'n': 10, 'days': 1}
# retest
r_train = objective(**best_params, crsp_data=sec, sig_rank_selected='high', train=True, report=True, filename='report_train.html', trading_cost=True)
r_test = objective(**best_params, crsp_data=sec, sig_rank_selected='high', train=False, report=True, filename='report_test.html', trading_cost=True)
# compare with SPY
SPY = utils.download_return('2010-01-01', '2020-12-31', 'SPY').set_index('Date')['r']
r_port = r_test.loc[r_test['sig_rank'] == 'high'].reset_index()
r_port['SPY'] = SPY
r_port['diff'] = r_port['forward_r'] - r_port['trading_cost'] - r_port['SPY']
r_port.plot(x='date', y='diff', kind ='bar')
plt.show()
