import numpy as np

regressor_config_dict = {

    'sklearn.ensemble.ExtraTreesRegressor': {
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
    },

    'sklearn.ensemble.AdaBoostRegressor': {
    },

    'sklearn.tree.DecisionTreeRegressor': {
    },

    'sklearn.neighbors.KNeighborsRegressor': {
    },

    'sklearn.svm.LinearSVR': {
    },

    'sklearn.ensemble.RandomForestRegressor': {
    },


    'xgboost.XGBRegressor': {
    },

    # Preprocessors
    'sklearn.preprocessing.Binarizer': {
    },

    'sklearn.decomposition.FastICA': {
    },

    'sklearn.cluster.FeatureAgglomeration': {
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
    },

    'sklearn.decomposition.PCA': {
    },

    'sklearn.preprocessing.PolynomialFeatures': {
    },

    'sklearn.kernel_approximation.RBFSampler': {
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
    },


    # Selectors
    'sklearn.feature_selection.SelectFwe': {
    },

    'sklearn.feature_selection.SelectPercentile': {
    },

    'sklearn.feature_selection.VarianceThreshold': {
    },

    'sklearn.feature_selection.SelectFromModel': {
    }

}
