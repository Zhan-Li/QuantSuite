"""
custom scoring functions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from quantsuite import get_port_r


def _sharpe_ratio(r_true: pd.Series, r_pred: pd.Series):
    port_r = get_port_r(r_true, r_pred, ntile=10)
    return port_r.mean() / port_r.std()


def _IC(r_true, r_pred):
    cov = np.cov(r_true, r_pred)
    if cov[0, 0] == 0 or cov[1, 1] == 0:
        return -2
    else:
        return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])


sharpe_ratio = make_scorer(_sharpe_ratio, greater_is_better=True)
IC = make_scorer(_IC, greater_is_better=True)
