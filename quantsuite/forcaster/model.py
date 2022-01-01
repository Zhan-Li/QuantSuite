from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class WeightedAverage(BaseEstimator, ClassifierMixin):

    def __init__(self, weight=None, thresh=None):
        """

        Parameters
        ----------
        weight : only support 'IC' and None temporarily. If weight is None, then use simple average.
        thresh : weight = 0 if abs(weight) <= thresh
        """
        self.thresh = thresh
        self.weight = weight

    def fit(self, X, y):
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError('y needs to have datetimeindex')
        coef = []
        if hasattr(X, 'columns'):
            n_cols = len(X.columns)
        elif hasattr(X, 'shape'):
            n_cols = X.shape[1]
        if not self.weight:
            self.coef_ = np.ones(n_cols)/sum(np.ones(n_cols))
            return self
        for i in range(n_cols):
            temp = y.to_frame(name='y')
            if type(X) == np.ndarray:
                temp['x'] = X[:, i]
            elif type(X) == pd.DataFrame:
                temp['x'] = X.iloc[:, i]
            temp = temp.dropna()
            if self.weight == 'IC':
                res = temp.groupby(temp.index).apply(lambda x: x.corr().iloc[0, 1]).mean()
            if abs(res) <= self.thresh:
                coef.append(0)
            else:
                coef.append(res)
        self.coef_ = np.array(coef)
        return self

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1) / sum(np.abs(self.coef_))
