import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from pandas import DataFrame
from ray.tune.sklearn import TuneSearchCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_squared_error
from sqlalchemy import MetaData, Table
from tpot import TPOTRegressor

from quantsuite import PerformanceEvaluation, PortfolioAnalysis


class ReturnForecaster:
    """
    Wrapper around sklearn and tpot, tailored for time-series return prediction.
    """
    def __init__(self, data):
        """
        'Data needs to have time index and only contains y and x variables.'
        """
        self.data = data
        self.best_params = None
        self.best_pipeline = None
        self.best_score = None
        self.y_test_true = None
        self.y_test_pred = None
        self.test_score = None
        self.perf = None
        if isinstance(data.index, pd.DatetimeIndex) is not True:
            raise Exception('Data index is not DatetimeIndex')

    def split_train_test(self, train_size: float, y_var: str):
        """
        split train validation data
        split_time: time to split train and valiadation dataset
        """
        index = self.data.index.unique().sort_values()
        cut_date = index[int(train_size * len(index))]
        train = self.data.loc[self.data.index < cut_date]
        test = self.data.loc[self.data.index >= cut_date]
        return train.drop(y_var, axis=1), train[y_var], test.drop(y_var, axis=1), test[y_var]

    @staticmethod
    def gen_cv_index(train: DataFrame, window_size: int, window_step=1, moving_window: bool = True):
        """
        customized cross-validation index
        train: training dataset
        window: window size
        """
        if isinstance(train.index, pd.DatetimeIndex) is not True:
            raise Exception('Data index is not DatetimeIndex')
        time = train.index.name
        train = train.reset_index()
        groups = train.groupby(time).groups
        sorted_index = [index.values for (key, index) in sorted(groups.items())]
        counter = range(window_size, len(groups) - window_size - window_step, window_step)
        if moving_window is True:
            cv = [(np.concatenate(sorted_index[i:i + window_size]),
                   np.concatenate(sorted_index[i + window_size:i + window_size + window_step]))
                  for i in counter]
        else:
            cv = [(np.concatenate(sorted_index[0:i + window_size]),
                   np.concatenate(sorted_index[i + window_size:i + window_size + window_step]))
                  for i in counter]
        return cv

    def search(self, params, pipeline, x_train, y_train, cv_train, n_trial, n_jobs=-1, use_gpu=False, verbose=2):
        """
        search hyperparameter with the training and validation data and predict with the test dataset.
        """
        search = TuneSearchCV(
            estimator=pipeline,
            param_distributions=params,
            search_optimization="hyperopt",
            n_trials=n_trial,
            early_stopping=False,
            scoring='neg_mean_squared_error',
            cv=cv_train,
            max_iters=1,
            verbose=verbose,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
        )
        print('Searching the the best hyperparameters...')
        search.fit(x_train, y_train)
        self.best_score = search.best_score_
        self.best_params = search.best_params_
        self.best_pipeline = search.best_estimator_

    def autoML_tpot(self, x_train, y_train, cv, use_gpu=True, generations=100, max_time_mins=None,
                    max_eval_time_mins=30, save_pipeline=False, file_name=None):
        """auto-ml with tpot"""
        if use_gpu is True:
            config_dict = 'TPOT cuML'
            n_jobs = 1
        else:
            config_dict = None
            n_jobs = -1
            print("Warning, you're not using GPU.")

        tpot = TPOTRegressor(generations=generations, population_size=100, scoring='neg_mean_squared_error',
                             verbosity=2,
                             cv=cv, n_jobs=n_jobs, max_time_mins=max_time_mins,
                             max_eval_time_mins=max_eval_time_mins, use_dask=True,
                             early_stop=10, memory='auto', random_state=0, config_dict=config_dict)
        print('Start autoML with Tpot...')
        tpot.fit(x_train.values, y_train.values.reshape(-1, ))
        if save_pipeline:
            if file_name is not None:
                tpot.export('fitted.py')
            else:
                raise Exception('File name not given for the exporeted pipeline')
        return tpot.fitted_pipeline_

    def interpret(self, X, y, feature_names, n_repeats):
        """model interpretation with partial dependence and permutation importance"""
        def plot_partial_dependence():
            fig, ax = plt.subplots(figsize=(12, len(feature_names) // 3 * 4))
            PartialDependenceDisplay.from_estimator(self.best_pipeline, X, range(len(feature_names)),
                                                    feature_names=feature_names,
                                                    n_jobs=-1, verbose=10, ax=ax)
            fig.subplots_adjust(hspace=0.3)
            fig.tight_layout()

        def plot_permutation_importance():
            result = permutation_importance(self.best_pipeline, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=0)
            mean = pd.Series(result.importances_mean, index=feature_names)
            fig, ax = plt.subplots()
            mean.sort_values().plot.barh(xerr=result.importances_std, ax=ax)
            ax.set_title("Permutation Importance")
            ax.set_xlabel("Mean increase in metric")
            fig.tight_layout()

        return plot_partial_dependence(), plot_permutation_importance()

    def backtest(self, spark, X, y, cv, time:str, freq: str, model = None):
        """
        backtest equal weighted long-short portfolio performance on the test dataset

        Parameters
        ----------
        spark: pyspark object
        X : features
        y : target
        cv : index for train, predict.
        model : model has higher priority than self.best_pipeline. That is, If model and fitted_best_pipeline are both
            supplied, fitted_best pipeline will be used instead. If model is None, then fitted best_pipeline
            will be used. Note that best_pipeline from automl_tpot cannot be used because tpot pipeline does not include
            tranformers.

        Returns
        -------

        """
        print('Predicting on the test set...')

        if model:
            estimator = model
        elif self.best_pipeline:
            estimator = self.best_pipeline
        else:
            raise Exception('Either model needs to be supplied or best_pipeline needs to provided by search')

        self.y_test_true = pd.concat([y.iloc[predict_index] for train_index, predict_index in cv])
        y_test_pred = [estimator.fit(X.iloc[train_index], y.iloc[train_index]).predict(X.iloc[predict_index])
                       for train_index, predict_index in cv]
        self.y_test_pred = pd.Series(np.concatenate(y_test_pred), index=self.y_test_true.index)
        self.test_score = -mean_squared_error(self.y_test_true, self.y_test_pred)

        returns = pd.DataFrame({'y_true': self.y_test_true, 'y_pred': self.y_test_pred}).reset_index()
        returns = spark.createDataFrame(returns).withColumn('name', f.lit('None'))
        pa = PortfolioAnalysis(returns, time, 'name', 'y_pred', 'y_true')
        pa.gen_portr(1, 10)
        r = pa.portr
        r = r.loc[r['sig_rank'] == 'high_minus_low']['y_true']
        perf = PerformanceEvaluation(r)
        self.perf = perf.get_stats(freq)


    def insert_to_db(self, dataset_names: List[str], sample_start: str, sample_end: str, sample_freq: str,
                     model_name: str, table: str, db_con):
        """
        Insert search result and performance statistics into dataset
        """
        if None in [self.best_score, self.best_params, self.pipeline, self.test_score, self.perf]:
            raise Exception('Need to search the the best hyperparameters/examine performance before insertion.')
        record = {'dataset_name': ','.join(dataset_names), 'sample_start': sample_start, 'sample_end': sample_end,
                  'sample_freq': sample_freq, 'model_name': model_name,
                  'best_val_score': self.best_score, 'test_score': self.test_score,
                  'best_params': self.best_params,
                  'perf': self.perf,
                  'best_model': pickle.dumps(self.pipeline)}
        dtypes = ['VARCHAR(100)', 'VARCHAR(20)', 'VARCHAR(20)', 'VARCHAR(10)', 'VARCHAR(60)', 'DOUBLE', 'DOUBLE',
                  'JSON', 'JSON', 'LONGBLOB']

        create_table_command = f'CREATE TABLE IF NOT EXISTS {table} (id INT NOT NULL AUTO_INCREMENT, ' \
                               f'{", ".join([x + " " + y for x, y in zip(record.keys(), dtypes)])}, ' \
                               'PRIMARY KEY (id));'

        db_con.execute(create_table_command)
        db_con.execute(
            Table(table, MetaData(), autoload_with=db_con).insert(),
            [record]
        )
        print(f'Sucessfully inserted records into table {table}')


def plot_feature_importances(trees, feature_names):
    importance = [tree.feature_importances_ for tree in trees]
    mean, std = np.mean(importance, axis=0), np.std(importance, axis=0)
    mean = pd.Series(mean, index=feature_names)
    fig, ax = plt.subplots()
    mean.sort_values().plot.barh(xerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_xlabel("Mean decrease in metric")
    fig.tight_layout()
