import argparse
import os

from filelock import FileLock
from tensorflow.keras.datasets import mnist

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from quantsuite.forcaster import Pipe
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)




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
preprocessor.fit(train_features)
train_features = preprocessor.transform(train_features)
test_features = preprocessor.transform(test_features)


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

config={
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": tune.uniform(0.001, 0.01),
                "momentum": tune.uniform(0.1, 0.9),
                "patience": tune.randint(10, 100),
                "n_layers": tune.randint(1,5),
                "n_hidden": tune.randint(32, 512),
                'dropout_rate': tune.uniform(0.1, 0.5)
            }

cv = list(KFold(n_splits=5, shuffle=True).split(train_features, train_labels))


class TensorFlowForcaster:

    def __init__(self):
        pass

    def __train(self, X, y, cv, config, transfomer=None):
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
                      optimizer=tf.keras.optimizers.SGD(learning_rate=config['learning_rate'], momentum=config['momentum']),
                      metrics=["mse"])
        # train
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=config['patience'])
        val_scores = []
        ys = []
        for train_idx, val_idx in cv:
            # transorm data
            if not transfomer:
                transfomer.fit(X[train_idx])
                x_train = transfomer.transform(X[train_idx])
            else:
                x_train = X[train_idx]
            model.fit(
                x_train,
                y.values[train_idx],
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                verbose=0,
                validation_split=0.2,
                callbacks=[es])
            y = train_labels.iloc[val_idx].to_frame(name='y_true')
            y['y_pred'] = model(train_features[val_idx]).numpy()
            ys.append(y)

            m = tf.keras.metrics.MeanSquaredError()
            m.update_state(y['y_true'], y['y_pred'])
            val_scores.append(m.result().numpy())
            avg_val_score = sum(val_scores) / len(val_scores)
            return avg_val_score, pd.concat(ys, axis=0)


    def __objective(self, config):
        avg_val_score, _= self.__train(config)
        tune.report(best_val_score =avg_val_score)


    def search(self, params, n_paralles, n_trial, verbose = 2):
        analysis = tune.run(
            self.__objective,
            metric="best_val_score",
            mode="min",
            num_samples=n_trial,
            resources_per_trial={
                "cpu": 0,
                "gpu": 1/n_paralles
            },
            verbose=verbose,
            config=params)
        return analysis

if __name__ == '__main__':
    res = TensorFlowForcaster(train_features, train_labels, cv).search(config, 20, 100)

    res.results
    res.best_config
    res.best_result
