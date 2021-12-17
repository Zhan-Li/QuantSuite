
import numpy as np

regressor_config_cuml = {
    # cuML + DMLC/XGBoost Regressors removing linear models
    "cuml.neighbors.KNeighborsRegressor": {},
    "xgboost.XGBRegressor": {},

    # Sklearn Preprocesssors

    "sklearn.preprocessing.Binarizer": {
    },

    "sklearn.decomposition.FastICA": {
    },

    "sklearn.cluster.FeatureAgglomeration": {
    },

    "sklearn.preprocessing.MaxAbsScaler": {
    },

    "sklearn.preprocessing.MinMaxScaler": {
    },

    "sklearn.preprocessing.Normalizer": {
    },

    "sklearn.kernel_approximation.Nystroem": {
    },

    "sklearn.decomposition.PCA": {
    },

    'sklearn.preprocessing.PolynomialFeatures': {
    },

    "sklearn.kernel_approximation.RBFSampler": {
    },

    "sklearn.preprocessing.RobustScaler": {
    },

    "sklearn.preprocessing.StandardScaler": {
    },

    "tpot.builtins.ZeroCount": {
    },

    "tpot.builtins.OneHotEncoder": {
    },

    # Selectors

    "sklearn.feature_selection.SelectFwe": {},

    "sklearn.feature_selection.SelectPercentile": {},

    "sklearn.feature_selection.VarianceThreshold": {},
    'sklearn.feature_selection.SelectFromModel': {
    }
}
