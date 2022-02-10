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

from quantsuite.forcaster.scorer import IC
from quantsuite.performance_evaluation import PerformanceEvaluation
from quantsuite.portfolio_analysis import PortfolioAnalysis
from ray.tune.schedulers import ASHAScheduler
from keras.callbacks import EarlyStopping
from ray import tune


def plot_feature_importances(trees, feature_names):
    importance = [tree.feature_importances_ for tree in trees]
    mean, std = np.mean(importance, axis=0), np.std(importance, axis=0)
    mean = pd.Series(mean, index=feature_names)
    fig, ax = plt.subplots()
    mean.sort_values().plot.barh(xerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_xlabel("Mean decrease in metric")
    fig.tight_layout()


class TrainTestSplitter:
    """Split time series into train test datasets"""

    def __init__(self, data: DataFrame):
        """
              split train validation data
              train_size: fraction to used as train
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise Exception('Data index needs to be pd.DatetimeIndex')
        self.data = data

    def split(self, test_offset_periods, train_end_time=None, train_fraction=None):
        """

        Parameters
        ----------
        test_offset_periods : int. if test_offset_periods is negative, then train_end_time-test_offset_periods to train_
        end_time will serve as the training data for testing. For example, test_offset_periods = -10, then last 10
        periods will be used as training data.
        train_end_time : str or None, end cutoff time for training data
        train_fraction : float or None, fraction of data as training data

        Returns: tuple of pandas DataFrames
        -------

        """
        if train_fraction and train_end_time:
            raise ValueError('Either train_end_time or train_fraction but not both.')
        if not train_fraction and not train_end_time:
            raise ValueError('Either train_end_time or train_fraction needs to be provided.')
        if not train_end_time:
            index = self.data.index.unique().sort_values()
            train_end_time = index[int(train_fraction * len(index))]
        test_start_date = self.data.index[self.data.index <= train_end_time].unique().sort_values()[test_offset_periods]
        return self.data.loc[self.data.index <= train_end_time], self.data.loc[
            self.data.index >= test_start_date]


class CustomCV:
    """"""

    def __init__(self, data: DataFrame):
        self.data = data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise Exception('Data index needs to be pd.DatetimeIndex')

    def gen_cv_index(self, train_window_size: int, val_window_size: int, window_step: int, moving_window: bool = True):
        """
        customized cross-validation index
        train: training dataset
        window_step: step between difference train windows. For example, if first train window starts at index 0
            then the second train window will start at index 10 if window_step = 10.
        """
        time = self.data.index.name
        train = self.data.reset_index()
        groups = train.groupby(time).groups
        sorted_index = [index.values for (key, index) in sorted(groups.items())]
        cv = []
        i = 0
        while i + train_window_size <= len(sorted_index) - 1:
            train_index = sorted_index[i if moving_window else 0: i + train_window_size]
            val_index = sorted_index[i + train_window_size: i + train_window_size + val_window_size]
            cv.append((np.concatenate(train_index), np.concatenate(val_index)))
            i = i + window_step
        return cv

    def gen_train_pred_index(self, train_window_size: int, pred_window_size: int, window_step,
                             moving_window: bool = True, print_index: bool = True):
        train_pred_index = self.gen_cv_index(train_window_size, pred_window_size, window_step, moving_window)
        if print_index:
            print(f'The first train index is: \n {self.data.iloc[train_pred_index[0][0]].index}')
            print(f'The first pred index is: \n {self.data.iloc[train_pred_index[0][1]].index}')
        return train_pred_index


class Forecaster:
    """
    Wrapper around sklearn and tpot, tailored for time-series return prediction.
    """
    scorers = {'IC': IC}

    def __init__(self, x_train, y_train, cv_train):
        self.x_train = x_train
        self.y_train = y_train
        self.cv_train = cv_train
        self.best_params = None
        self.best_pipeline = None
        self.best_score = None
        self.y_test_true = None
        self.y_test_pred = None
        self.test_score = None
        self.perf = None

    def search(self, params, pipeline, scoring, n_trial, max_search_seconds, n_jobs=-1, use_gpu=False,
               save_result=True, early_stopping=False, file_name='ray_best_pipeline.pkl',
               verbose=2):
        """
        ray tune search using scikit API
        Singple model hyperparameter tuner
        search hyperparameter with the training and validation data and predict with the test dataset.
        """
        search = TuneSearchCV(
            estimator=pipeline,
            param_distributions=params,
            search_optimization="hyperopt",
            n_trials=n_trial,
            early_stopping=ASHAScheduler() if early_stopping else None,
            scoring=self.scorers[scoring] if type(scoring) == str else scoring,
            cv=self.cv_train,
            max_iters=100 if early_stopping else 1,
            verbose=verbose,
            n_jobs=n_jobs,
            time_budget_s=max_search_seconds,
            use_gpu=use_gpu,
        )
        print('Searching the the best hyperparameters...')
        search.fit(self.x_train, self.y_train)
        self.best_score = search.best_score_
        self.best_params = search.best_params_
        self.best_pipeline = search.best_estimator_
        if save_result:
            with open(file_name, 'wb') as file:
                pickle.dump(search.best_estimator_, file)

    def search_tf(self):
        return

    def autoML_tpot(self, config_dict, n_jobs, generations=100, scoring='neg_mean_squared_error', max_time_mins=None,
                    max_eval_time_mins=30, save_pipeline=False, file_name=None):
        """auto-ml with tpot."""
        if save_pipeline and not file_name:
            raise Exception('File name not given for the exporeted pipeline')

        tpot = TPOTRegressor(generations=generations, population_size=100, scoring=scoring,
                             verbosity=2,
                             cv=self.cv_train, n_jobs=n_jobs, max_time_mins=max_time_mins,
                             max_eval_time_mins=max_eval_time_mins, use_dask=False,
                             early_stop=10, memory='auto', config_dict=config_dict)
        print('Start autoML with Tpot...')
        x_train = self.x_train.values if hasattr(self.x_train, 'values') else self.x_train
        y_train = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train

        tpot.fit(x_train, y_train.reshape(-1, ))
        if save_pipeline:
            tpot.export(file_name)
        return tpot.fitted_pipeline_

    def interpret(self, X, y, feature_names, n_repeats):
        """model interpretation with partial dependence and permutation importance."""

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

    def predict(self, spark, X, y, train_predict_index, time: str, freq: str, model=None):
        """
        backtest equal weighted long-short portfolio performance on the test dataset.

        Parameters
        ----------
        spark: pyspark object
        X : features
        y : target, pd.series with time index
        train_predict_index : index for train, predict.
        model : model has higher priority than self.best_pipeline. That is, If model and fitted_best_pipeline are both
            supplied, fitted_best pipeline will be used instead. If model is None, then fitted best_pipeline
            will be used. Note that best_pipeline from automl_tpot cannot be used because tpot pipeline does not include
            tranformers.

        Returns
        -------

        """
        print('Backtesting...')

        if model:
            estimator = model
        elif self.best_pipeline:
            estimator = self.best_pipeline
        else:
            raise Exception('Either model needs to be supplied or best_pipeline needs to provided by search')

        self.y_test_true = pd.concat([y.iloc[predict_index] for train_index, predict_index in train_predict_index])

        if type(X) == np.ndarray:
            y_test_pred = [estimator.fit(X[train_index], y.iloc[train_index]).predict(X[predict_index])
                           for train_index, predict_index in train_predict_index]
        elif type(X) == pd.DataFrame:
            y_test_pred = [estimator.fit(X.iloc[train_index], y.iloc[train_index]).predict(X.iloc[predict_index])
                           for train_index, predict_index in train_predict_index]
        else:
            raise TypeError('X is either numpy.ndarray or pd.DataFrame')
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
        self.port_r = r

    def insert_to_db(self, dataset_names: List[str], sample_start: str, sample_end: str, sample_freq: str,
                     model_name: str, table: str, db_con):
        """
        Insert search result and performance statistics into dataset.
        """
        if None in [self.best_score, self.best_params, self.best_pipeline, self.test_score, self.perf]:
            raise Exception('Need to search the the best hyperparameters/examine performance before insertion.')
        record = {'dataset_name': ','.join(dataset_names), 'sample_start': sample_start, 'sample_end': sample_end,
                  'sample_freq': sample_freq, 'model_name': model_name,
                  'best_val_score': self.best_score, 'test_score': self.test_score,
                  'best_params': self.best_params,
                  'perf': self.perf,
                  'best_model': pickle.dumps(self.best_pipeline)}
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


class TensorFlowForcaster:

    def __init__(self, transformer=None):
        self.transformer = transformer

    def __train(self, config):
        # get data
        X, y, cv = config['X'], config['y'], config['cv']
        # set GPU memory
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # build model
        model = tf.keras.Sequential()
        for i in range(config['n_layers']):
            model.add(tf.keras.layers.Dense(units=config['n_hidden'], activation='relu'))
            model.add(tf.keras.layers.Dropout(config['dropout_rate']))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],
                                                        momentum=config['momentum']),
                      metrics=["mse"])
        # train
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=config['patience'])
        val_scores = []
        ys = []
        for train_idx, val_idx in cv:
            # transorm data
            if self.transformer:
                self.transformer.fit(X.iloc[train_idx])
                x_train = self.transformer.transform(X.iloc[train_idx])
                x_val = self.transformer.transform(X.iloc[val_idx])
            else:
                x_train = X.iloc[train_idx]
                x_val = X.iloc[val_idx]

            model.fit(
                x_train,
                y.iloc[train_idx],
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                verbose=0,
                # validation_split= 0.2,
                validation_data=(x_val, y.iloc[val_idx]),
                callbacks=[es])
            y_slice = y.iloc[val_idx].to_frame(name='y_true')
            y_slice['y_pred'] = model(x_val).numpy()
            ys.append(y_slice)

        m = tf.keras.metrics.MeanSquaredError()
        m.update_state(y_slice['y_true'], y_slice['y_pred'])
        val_scores.append(m.result().numpy())
        avg_val_score = sum(val_scores) / len(val_scores)
        return avg_val_score, pd.concat(ys, axis=0)

    def __objective(self, config):
        avg_val_score, _ = self.__train(config)
        tune.report(best_val_score=avg_val_score)

    def search(self, X, y, cv, params, n_trial, n_paralles=10, verbose=2):
        params['X'], params['y'], params['cv'] = X, y, cv
        analysis = tune.run(
            self.__objective,
            metric="best_val_score",
            mode="min",
            num_samples=n_trial,
            resources_per_trial={
                "cpu": 0,
                "gpu": 1 / n_paralles
            },
            verbose=verbose,
            config=params)
        return analysis

    def predict(self, X, y, train_pred_split, params):
        params['X'], params['y'], params['cv'] = X, y, train_pred_split
        avg_score, ys = self.__train(params)
        return avg_score, ys



if __name__ == '__main__':
    from sklearn.model_selection import KFold
    from quantsuite.forcaster import Pipe

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    preprocessor = Pipe(model_str='None', num_cols=train_features.columns.to_list()).preprocess()

    # K-fold Cross Validation model evaluation

    config = {
        "batch_size": 32,
        "epochs": 100,
        "lr": 0.01,
        "momentum": 0.1,
        "patience": 10,
        "n_layers": 2,
        "n_hidden": 32,
        'dropout_rate': 0.1
    }

    config = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": tune.uniform(0.001, 0.01),
        "momentum": tune.uniform(0.1, 0.9),
        "patience": tune.randint(10, 100),
        "n_layers": tune.randint(1, 5),
        "n_hidden": tune.randint(32, 512),
        'dropout_rate': tune.uniform(0.1, 0.5)
    }

    cv_train = list(KFold(n_splits=5, shuffle=True).split(train_features, train_labels))
    cv_test = list(KFold(n_splits=5, shuffle=True).split(test_labels, test_labels))
    TFF = TensorFlowForcaster(transformer=preprocessor)
    res = TFF.search(train_features, train_labels, cv_train, config, 10, 10, verbose = 2)
    TFF.predict(test_features, test_labels, cv_test, res.best_config)
