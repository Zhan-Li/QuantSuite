from sklearn.base import TransformerMixin


class Winsorizer(TransformerMixin):
    """
    Winsorizing at the threshold. Can only be applied after standardization.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X[X > self.threshold] = self.threshold
        X[X < -self.threshold] = -self.threshold
        return X


class Identity(TransformerMixin):
    """
    Identity transforamtion, i.e., the data stays the same
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X

    def inverse_transform(self, X, y=None, **fit_params):
        return X
