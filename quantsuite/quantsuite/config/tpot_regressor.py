import numpy as np

regressor_config = {

    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0],
        'objective': ['reg:squarederror']
    },

    # Preprocessors
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },


    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}
regressor_config_cuml = {
    # cuML + DMLC/XGBoost Regressors
    "cuml.neighbors.KNeighborsRegressor": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform"],
    },

    "cuml.ensemble.RandomForestRegressor": {
        "n_estimators": [50, 100, 150],
        "max_depth": range(10, 100, 5),
        "max_features": np.arange(0.1, 1, 0.1),
        "n_bins": range(100, 1000, 100),
    },

    "xgboost.XGBRegressor": {
        "n_estimators": [50, 100, 150],
        "max_depth": range(3, 10),
        "learning_rate": [1e-2, 1e-1, 0.5, 1.],
        "min_child_weight": range(1, 21),
        "subsample": np.arange(0.1, 1, 0.1),
        "colsample_bytree": np.arange(0.1, 1, 0.1),
        "tree_method": ["gpu_hist"],
        "n_jobs": [1],
        "verbosity": [0],
        "objective": ["reg:squarederror"]
    },

    # Sklearn Preprocesssors

    "sklearn.decomposition.FastICA": {
        "tol": np.arange(0.0, 1.01, 0.05)
    },

    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"]
    },

    "sklearn.preprocessing.MaxAbsScaler": {
    },

    "sklearn.preprocessing.MinMaxScaler": {
    },

    "sklearn.preprocessing.Normalizer": {
        "norm": ["l1", "l2", "max"]
    },

    "sklearn.kernel_approximation.Nystroem": {
        "kernel": ["rbf", "cosine", "chi2", "laplacian", "polynomial", "poly", "linear", "additive_chi2", "sigmoid"],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11)
    },

    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11)
    },

    "sklearn.kernel_approximation.RBFSampler": {
        "gamma": np.arange(0.0, 1.01, 0.05)
    },

    "sklearn.preprocessing.RobustScaler": {
    },

    "sklearn.preprocessing.StandardScaler": {
    },

    "tpot.builtins.ZeroCount": {
    },

    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10]
    },

    # Selectors

    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {
            "sklearn.feature_selection.f_classif": None
        }
    },

    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {
            "sklearn.feature_selection.f_classif": None
        }
    },

    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    }
}
