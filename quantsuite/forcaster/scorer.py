from sklearn.metrics import make_scorer
import pandas as pd


def get_port_r(r_true: pd.Series, r_pred: pd.Series):
    if not isinstance(r_true.index, pd.DatetimeIndex):
        raise Exception('r_true needs to have datatime index')
    returns = r_true.to_frame(name='r_true')
    returns['r_pred'] = r_pred
    returns['r_pred_pct'] = returns['r_pred'].groupby(returns.index).rank(method='first', pct=True)
    return returns.groupby(returns.index) \
        .apply(
        lambda x: x.loc[x['r_pred_pct'] >= 0.9]['r_true'].mean() - x.loc[x['r_pred_pct'] <= 0.1]['r_true'].mean())


def _get_sharpe_ratio(r_true, r_pred):
    port_r = get_port_r(r_true, r_pred)
    return port_r.mean() / port_r.std()


get_sharpe_ratio = make_scorer(_get_sharpe_ratio, greater_is_better=True)
