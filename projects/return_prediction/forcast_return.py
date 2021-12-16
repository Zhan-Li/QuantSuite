from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor

from xgboost import XGBRegressor
import importlib
from quantsuite.forcaster import ReturnForecaster, CustomCV, TrainTestSplitter
from quantsuite import misc_funcs
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
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
import json
from sqlalchemy import create_engine, insert, Table, MetaData
from quantsuite.transformers import Winsorizer, Identity

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
start = '2008-06-01'
start_year = 2008
freq = 'daily'
data_file ='data/return_prediction.pkl'
forward_r = 'forward_r_d'
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin= secrets['mysql_password']
check_data = False
tune_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/tune_result')
# read data
stock = pd.read_pickle(data_file)
stock = stock.loc[stock['date'] >= start]
# preprocess data
stock['rank'] = stock.groupby('date')['mktcap'].rank(ascending=False)
stock = stock.loc[stock['rank'] <= 500].drop(['rank', 'mktcap'], axis = 1)
stock[forward_r] = stock.sort_values('date').groupby('cusip')['ret'].shift(-1)
stock = stock.drop(['cusip', 'ret'], axis = 1)
na_removed = misc_funcs.select_cols_by_na(stock, 0.3)
corr_removed = misc_funcs.select_cols_by_corr(na_removed, forward_r, na_removed.iloc[:, 3:].columns, 0.5)
mydata= corr_removed.sort_values('date').set_index('date').dropna(subset = [forward_r])
if check_data is True:
    examine_res = pd.DataFrame()
    examine_res = examine_res.append(misc_funcs.analyze_data(stock, 'IQR', 1.5))
    examine_res.to_html('data_report.html')


# build pipelne
num_cols = mydata.iloc[:, :-1].columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scale', RobustScaler()),
    ('cutoff', Winsorizer(1.5)),  # Cut off at 1.5 IQR)
])

preprocessor_x = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols)
    ])
preprocessor_y = Pipeline(steps=[
    ('copy', Identity())
])
# Initilize class
train,  test = TrainTestSplitter(mydata).split(1/(2020-start_year))
cv_train = CustomCV(train).gen_cv_index(10, 10)
cv_test = CustomCV(test).gen_cv_index(10, 10)
x_train, y_train = train.drop([forward_r], axis = 1), train[forward_r]
x_test, y_test = test.drop([forward_r], axis = 1), test[forward_r]

x_train_transformed = preprocessor_x.fit_transform(x_train)
# auto-ML
rf = ReturnForecaster(x_train_transformed, y_train, cv_train)
#res = rf.autoML_tpot(100, 12*60,  save_pipeline=True, file_name='fitted.py', use_gpu=False)
# train with model from auto_ML-----------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_x),
    ('model', SGDRegressor(penalty='elasticnet', learning_rate = 'invscaling'))
])
estimator = TransformedTargetRegressor(regressor=pipeline, transformer=preprocessor_y)
# train
params ={'regressor__model__alpha': tune.uniform(0, 1),
        'regressor__model__l1_ratio': tune.uniform(0, 1),
        'regressor__model__max_iter': tune.qrandint(1000, 10000, 1000),
        'regressor__model__tol': tune.uniform(0, 0.1),
         'regressor__model__eta0': tune.uniform(0, 1),
         'regressor__model__power_t': tune.quniform(0.1, 150.0, 0.1),
         'regressor__model__early_stopping': tune.choice([True, False])
    }
rf = ReturnForecaster(x_train, y_train, cv_train)
rf.search(params, estimator, 100)
rf.backtest(spark, x_train, y_train, cv_train, 'date', 'daily')
rf.perf
# ANN------------------------------------

random_grid ={
        'n_layers': tune.qrandint(1, 10, 1),
        'n_units': tune.qrandint(32, 128, 16),
        'dropout_layer': tune.choice([True, False]),
        'drop_rate': tune.sample_from(lambda spec: tune.quniform(0, 0.9, 0.1) if spec.config.dropout_layer == True else 0),
        'adam_rate': tune.quniform(0.001, 0.01, 0.001),
    }






from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=create_baseline, epochs=10000, batch_size=1000,  verbose=0)
results = cross_val_score(estimator, x_train_transformed, y_train, cv=cv_train, verbose=10)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



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



