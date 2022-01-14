from ray import tune
from sklearn.ensemble import *
from quantsuite.forcaster.model import WeightedAverage
from sklearn.tree import *
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

    'weighted_average':{
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
}
