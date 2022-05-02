from ray import tune
from sklearn.ensemble import *
from quantsuite.forcaster.model import WeightedAverage
from sklearn.tree import *
from xgboost.sklearn import XGBRegressor

auto_ml_config = {
    'random_forest_regressor': {
        'params': {'n_estimators': tune.qrandint(50, 150, 50),
                   'max_features': tune.quniform(0.05, 1.0, 0.05),
                   'min_samples_split': tune.randint(2, 21),
                   'min_samples_leaf': tune.randint(1, 21),
                   'bootstrap': tune.choice([True, False])
                   },
        'model': RandomForestRegressor()
    },

    'weighted_average': {
        'params': {'thresh': tune.quniform(0.001, 0.02, 0.001)
                   },
        'model': WeightedAverage()},

    'ada_boost_regressor': {
        'params': {'base_estimator': tune.choice([DecisionTreeRegressor(max_depth=n) for n in range(2, 7)]),
                   'n_estimators': tune.qrandint(50, 150, 50),
                   'learning_rate': tune.quniform(0.05, 1.0, 0.05),
                   'loss': tune.choice(['linear', 'square', 'exponential'])
                   },
        'model': AdaBoostRegressor()
    },
    'xgb_regressor': {
        'params':
            {'n_estimators': [100],
             'max_depth': tune.randint(1, 11),
             'learning_rate': tune.choice([1e-3, 1e-2, 1e-1, 0.2, 0.5, 0.7, 1.0]),
             'gamma': tune.uniform(0, 1),
             'min_child_weight': tune.randint(1, 50),
             'subsample': tune.quniform(0.1, 1, 0.1),
             'colsample_bytree': tune.quniform(0.1, 1, 0.1),
             'n_jobs': [1],
             'verbosity': [0],
             'objective': ['reg:squarederror']},
        'model': XGBRegressor()
    },
    'tf_regressor': {
        'params': {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": tune.quniform(0.005, 0.1, 0.005),
            "momentum": tune.quniform(0, 0.1, 0.01),
            "patience": tune.choice([100]),
            "n_layers": tune.randint(1, 5),
            "n_hidden": tune.qrandint(32, 32*2, 32),
            'dropout_rate': tune.quniform(0, 0.5, 0.1)
        },
        'model': 'None'
    }

}
