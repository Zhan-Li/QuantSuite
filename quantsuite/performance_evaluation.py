import pandas as pd
import numpy as np
import seaborn as sns
import quantsuite.misc_funcs as utils
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.linear_model import OLS


class PerformanceEvaluation:
    """
    The OptionSignal class gnerates option-based trading signals.
    """

    def __init__(self, r:pd.Series):
        if isinstance(r.index, pd.DatetimeIndex) is not True:
            raise Exception('Need time series data')
        self.r = r

    def plot_wealth(self,  benchmark: pd.Series = None, title = 'Cumulative Returns'):
        if benchmark is None:
            return sns.lineplot(data= (self.r + 1).cumprod())
        else:
            returns = self.r.to_frame()
            returns[benchmark.name] = benchmark
            returns[self.r.name] = (returns[self.r.name]+1).cumprod()
            returns[benchmark.name] = (returns[benchmark.name]+ 1).cumprod()
            return sns.lineplot(data= returns)

    def plot_drawdowns(self):
        drawdowns = (1 + self.r).cumprod() / (1 + self.r).cumprod().cummax() - 1
        return sns.lineplot(data=drawdowns)

    def get_stats(self, freq: str, rf=0):
        mults = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly':4,  'yearly': 1}
        mult = mults[freq]
        excess_r = self.r - rf

        perfs = {}
        perfs['Average Return'] = self.r.mean()
        perfs['Max Return'] = self.r.max()
        perfs['Min Return'] = self.r.min()
        perfs['Max Drawdown'] = np.abs(((1 + self.r).cumprod() / (1 + self.r).cumprod().cummax() - 1).min())
        perfs['Volatility'] = self.r.std()
        perfs['Downside Deviation'] = self.r.loc[self.r < 0].std()
        perfs['95%-VaR'] = self.r.quantile(0.05)
        perfs['99%-VaR'] = self.r.quantile(0.01)
        perfs['95%-ES'] = self.r.loc[self.r <= self.r.quantile(0.05)].mean()
        perfs['99%-ES'] = self.r.loc[self.r <= self.r.quantile(0.01)].mean()
        perfs['Annualized Average Return'] = mult * perfs['Average Return']
        perfs['Annualized Geometric Return'] = (1 + self.r).cumprod().values[-1] ** (mult / len(self.r)) - 1
        perfs['Annual Volatility'] = np.sqrt(mult) * perfs['Volatility']
        perfs['Annual Downside Deviation'] = np.sqrt(mult) * perfs['Downside Deviation']
        perfs['Annual Sharpe Ratio'] = np.sqrt(mult) * excess_r.mean() / self.r.std()
        perfs['Annual Sortino Ratio'] = np.sqrt(mult) * excess_r.mean() / self.r.loc[self.r < 0].std()
        perfs['Annual Calmer Ratio'] = mult * excess_r.mean()/ perfs['Max Drawdown']
        perfs['Hit Rate'] = (self.r > 0).sum() / len(self.r)
        perfs['Loss Rate'] = (self.r < 0).sum() / len(self.r)
        perfs['Gross Profit'] = self.r.loc[self.r > 0].sum()
        perfs['Gross Loss'] = self.r.loc[self.r < 0].sum()
        perfs['Average Return in Up Days'] = self.r.loc[self.r > 0].mean()
        perfs['Average Return in Down Days'] = self.r.loc[self.r < 0].mean()
        perfs['Slugging Ratio'] = -1 * perfs['Average Return in Up Days'] / perfs['Average Return in Down Days']
        perfs['Profit Factor'] = -1 * perfs['Gross Profit'] / perfs['Gross Loss']

        return perfs

    def get_alpha(self, model='FF5', mom=True, freq='daily', window=None):
        """
        data: return of sorted portfolio returns per freq(daily, weekly, monthly, annually)
        model: 'FF3', 'FF5', 'q-factor', 'CAPM'
        """
        # download factors
        if model == 'FF3' or model == 'FF5':
            factors = utils.get_FF_factors(model, mom, freq)
        elif model == 'q-factor':
            factors = utils.get_q_factors(mom, freq)
        elif model == 'CAPM':
            factors = utils.get_FF_factors('FF3', False, freq)[['Mkt-RF', 'RF']]
        # get factor variables
        if mom is False:
            factor_cols = [i for i in factors.columns if i != 'Mom' and i != 'RF']
        elif mom is True:
            factor_cols = [i for i in factors.columns if i != 'RF']
        # match returns with factors
        returns = self.r
        date_format = {'daily': '%Y%m%d', 'weekly': '%Y%m%d', 'monthly': '%Y%m', 'quarterly': '%Y%m', 'yearly': '%Y'}
        returns.index = returns.index.strftime(date_format[freq])
        df = returns.to_frame(name = 'r').merge(factors, left_index=True, right_index=True, how='inner').dropna()
        df['r_excess'] = df['r'] - df['RF']
        # regression
        if window:
            res = RollingOLS(df['r_excess'], sm.add_constant(df[factor_cols]), window=window)\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        else:
            res = OLS(df['r_excess'], sm.add_constant(df[factor_cols])) \
                .fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        return res.params, res.tvalues, res.pvalues







