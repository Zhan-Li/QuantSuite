

from scipy.stats import randint, uniform
from sklearn.ensemble import AdaBoostRegressor
import pyspark.sql.functions as f
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import importlib
from invtools import PerformanceEvaluation, PortfolioAnalysis
import invtools.misc_funcs as misc_funcs
import class_return_forcaster; importlib.reload(class_return_forcaster)
from class_return_forcaster import ReturnForecaster
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, Dropout
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.optuna import OptunaSearch
import ray
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.keras import TuneReportCallback
import json
import random
import datetime
from sqlalchemy import create_engine, insert, Table, MetaData
import pickle
import time

sns.set_theme()
# params
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.maxResultSize', '2g')\
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# global params
start = '2020-01-01'
freq = 'daily'
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin= secrets['mysql_password']
check_data = False
tune_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/tune_result')
# preprocess data
stock = pd.read_pickle('data.pkl')

na_removed = misc_funcs.select_cols_by_na(stock, 0.3)
corr_removed = misc_funcs.select_cols_by_corr(na_removed, 'forward_ret', na_removed.iloc[:, 3:].columns, 0.5)
mydata= corr_removed.drop('cusip', axis = 1).sort_values('date').set_index('date')
if check_data is True:
    examine_res = pd.DataFrame()
    examine_res = examine_res.append(misc_funcs.analyze_data(df, 'IQR', 1.5))
    examine_res.to_html('data_report.html')

# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# auto-ML
rf.autoML_tpot(x_train, y_train, cv_train, True,  1, 12*60)
rf.predict(x_test, y_test, cv_test)
# build pipelne
num_cols = mydata.iloc[:, 1:].columns
class winsorize(TransformerMixin):

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X[X > self.threshold] = self.threshold
        X[X < -self.threshold] = -self.threshold
        return X

class copy(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X

    def inverse_transform(self, X, y=None, **fit_params):
        return X

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scale', RobustScaler()),
    ('cutoff', winsorize(1.5)),  # Cut off at 1.5 IQR)
])

preprocessor_x = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols)
    ])
preprocessor_y = Pipeline(steps=[
    ('copy', copy())
])
# Random Forest-----------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model', RandomForestRegressor( random_state = 0))
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# train
params= {'regressor__model__n_estimators': tune.randint(10, 500),
            'regressor__model__min_samples_leaf':  tune.randint(50, 1000),
         'regressor__model__max_features': tune.qrandint(1, len(num_cols), 1)}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=100)
rf.predict(x_test, y_test, cv_test)
rf.simulate(spark,'daily', 'date')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', 'randomforestRegressor',
                   'option_sig', tune_connection)


# Adaboost
model_name = 'AdaBoostRegressor'
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model', AdaBoostRegressor( random_state = 0))])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# model interpretability
models =  [DecisionTreeRegressor(max_depth=n) for n in [1, 2, 3, 4, 5]]
params= {'regressor__model__base_estimator': models,
    'regressor__model__n_estimators': tune.randint(10, 500),
            'regressor__model__learning_rate':  tune.quniform(0.01, 1, 0.01),
         'regressor__model__loss': ['linear', 'square', 'exponential']}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=100)
rf.predict(x_test, y_test, cv_test)
rf.simulate(spark,'daily', 'date')
rf.best_params['regressor__model__base_estimator_max_depth'] = 1
rf.best_params.pop('regressor__model__base_estimator')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', model_name,
                   'option_sig', tune_connection)

# XGboost
model_name = 'XGBRegressor'
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model', XGBRegressor(objective = 'reg:squarederror', n_jobs =1, tree_method = 'hist',  random_state = 0))
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# train
params= {'regressor__model__n_estimators': tune.randint(10, 500),
        'regressor__model__max_depth':  tune.randint(1, 5),
            'regressor__model__learning_rate':  tune.quniform(0.01, 1, 0.01),
         'regressor__model__booster': ['gbtree'],
        'regressor__model__tree_method': ['approx'],
        'regressor__model__subsample': tune.quniform(0.1, 1, 0.1),
        'regressor__model__colsample_bytree': tune.quniform(0.1, 1, 0.1),
        'regressor__model__reg_alpha': tune.quniform(0.01, 1, 0.01),
         'regressor__model__reg_lambda': tune.quniform(0.01, 1, 0.01)}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=1000)
rf.predict(x_test, y_test, cv_test)
rf.backtest(spark,'daily', 'date')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', model_name,
                   'option_sig', tune_connection)

# KNN----------------------------
model_name = 'KNN'
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model',  KNeighborsRegressor())
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# train
params= {'regressor__model__n_neighbors': tune.randint(1, 500),
        'regressor__model__weights': ['distance', 'uniform']}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=100)
rf.predict(x_test, y_test, cv_test)
rf.backtest(spark,'daily', 'date')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', model_name,
                   'option_sig', tune_connection)


# Elastic net ---------
model_name = 'ElasticNet'
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model',  ElasticNet())
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# train
params= {'regressor__model__alpha': tune.uniform(0.001, 20),
        'regressor__model__l1_ratio': tune.uniform(0.001, 1)}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=500)
rf.predict(x_test, y_test, cv_test)
rf.backtest(spark,'daily', 'date')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', model_name,
                   'option_sig', tune_connection)

# Linear-SVR----------
model_name = 'LinearSVR'
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model',  LinearSVR(random_state=0))
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# Initilize class
rf = ReturnForecaster(mydata.dropna(subset = ['forward_ret']))
x_train, y_train, x_test, y_test = rf.split_train_test(0.8, 'forward_ret')
cv_train = rf.gen_cv_index(x_train, 10, 10)
cv_test = rf.gen_cv_index(x_test, 10, 10)
# train
params= {'regressor__model__epsilon': tune.uniform(0, 0.1),
        'regressor__model__C': tune.uniform(0.001, 10),
         'regressor__model__loss':['epsilon_insensitive', 'squared_epsilon_insensitive']}
rf.search(params, estimator, x_train, y_train,  cv_train, n_trial=200)
rf.predict(x_test, y_test, cv_test)
rf.backtest(spark,'daily', 'date')
rf.insert_to_db(['crsp', 'optionmetrics'], '2020-01-01', '2020-12-31', 'daily', model_name,
                   'option_sig', tune_connection)



# ANN------------------------------------

random_grid ={
        'n_layers': tune.qrandint(1, 10, 1),
        'n_units': tune.qrandint(32, 128, 16),
        'dropout_layer': tune.choice([True, False]),
        'drop_rate': tune.sample_from(lambda spec: tune.quniform(0, 0.9, 0.1) if spec.config.dropout_layer == True else 0),
        'adam_rate': tune.quniform(0.001, 0.01, 0.001),
    }





def objective(n_layers: int, n_units:int, dropout_layer: bool, drop_rate: float, adam_rate:float):
    normalizer = preprocessing.Normalization(axis=-1)
    # normalizer.adapt(np.array(train_X_df))
    model = tf.keras.Sequential(normalizer)
    for i in range(n_layers):
        model.add(Dense(n_units, activation='relu'))
    if dropout_layer is True:
        model.add(Dropout(drop_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(adam_rate))
    history = model.fit(
        train_X_df,
        train_Y_df,
        validation_split=0.2,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=5, verbose=1)],
        verbose=1,
        epochs=100)
    return history



