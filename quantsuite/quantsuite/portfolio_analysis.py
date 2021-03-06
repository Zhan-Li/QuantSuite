import pandas as pd
import numpy as np
import statsmodels.api as sm
import importlib
import quantsuite.misc_funcs as utils

importlib.reload(utils)
import pyspark.sql.functions as f
from pyspark.sql import DataFrame as PysparkDataFrame
from pyspark.sql import Window
from scipy.stats import ttest_1samp
from pandas import DataFrame as PandasDataFrame
from pyspark.sql.types import StringType, IntegerType
from typing import List


def get_port_r(r_true: pd.Series, r_pred: pd.Series, ntile=10):
    if not isinstance(r_true.index, pd.DatetimeIndex):
        raise Exception('r_true needs to have datatime index')
    returns = r_true.to_frame(name='r_true')
    returns['r_pred'] = r_pred
    returns['r_pred_pct'] = returns['r_pred'].groupby(returns.index).rank(method='first', pct=True)
    return returns.groupby(returns.index) \
        .apply(
        lambda x: x.loc[x['r_pred_pct'] >= 1 - 1 / ntile]['r_true'].mean() - x.loc[x['r_pred_pct'] <= 1 / ntile][
            'r_true'].mean())


class PortfolioAnalysis:
    """
    data: pyspark dataframe with security returns and signals
    """

    def __init__(self, data: PysparkDataFrame, time: str, name: str, sig: str, forward_r: str,
                 weight=None, var_list=None):
        """
        r: forward return
        """
        self.time, self.r, self.sig, self.name = time, forward_r, sig, name
        # data = data[[time, name, sig, forward_r, weight]] if vw is True else data[[time, name, sig, r]]
        # self.data = data.loc[data[self.sig].notnull()].loc[data[self.sig] != np.nan]

        self.data = data.filter(f.col(self.sig).isNotNull()).filter(f.col(self.sig) != np.nan)
        if self.data.limit(1).count == 0:
            raise ValueError('Data is empty')
        if weight is None or weight == 'None':
            self.weight = 'weight'
            self.data = self.data.withColumn('weight', f.lit(1))
        else:
            self.weight = weight

        if var_list is None:
            self.var_list = [sig]
        else:
            self.var_list = var_list + [sig]
        # self.symbols_all = self.data[self.name].unique().tolist()
        self.portr = None

    def gen_portr(self, shift: int, ntile=None, sort_var2=None, ntile2=None, dependent_sort=True,
                  total_position=0, commission=0, turnover=0) -> PandasDataFrame:
        """
        ntile: n quantile portfolio formation
        cond_var: conditional variable for double sorting
        total_position: total $$ to invest
        commission: broker commission per transaction
        turnover: portfolio turover rate.
        Raw returns for equal??? and value-weighted portfolios by ntiles and freq (day, month, week, or year)
        """
        # generate rank for conditional var
        if sort_var2 is not None and ntile2 is not None:
            self.data = self.data.filter(f.col(sort_var2).isNotNull()).filter(f.col(sort_var2) != np.nan)
            cond_ranked = self.data.withColumn('cond_rank',
                                               f.ntile(ntile2).over(Window.partitionBy(self.time).orderBy(sort_var2))) \
                .withColumn('cond_rank', f.col('cond_rank').cast(IntegerType()))
            sig_group = [self.time, 'cond_rank'] if dependent_sort is True else self.time
            port_group = [self.time, 'sig_rank', 'cond_rank']
            merge_vars = [self.time, 'cond_rank']
        else:
            cond_ranked = self.data
            sig_group = self.time
            port_group = [self.time, 'sig_rank']
            merge_vars = [self.time]
        # generate sig ranks
        sig_ranked = cond_ranked \
            .withColumn('sig_rank', f.ntile(ntile).over(Window.partitionBy(sig_group).orderBy(self.sig))) \
            .withColumn('sig_rank', f.col('sig_rank').cast(IntegerType()))
        # generate portfolio returns
        agged = sig_ranked.groupBy(port_group) \
            .agg((f.sum(f.col(self.r) * f.col(self.weight)) / f.sum(f.col(self.weight))).alias(self.r),
                 f.collect_list(self.name).alias(self.name),
                 f.count(self.r).alias('n_assets'),
                 *[f.mean(x).alias(x) for x in self.var_list])
        # generate high minus low
        high = agged.filter(f.col('sig_rank') == ntile) \
            .withColumnRenamed(self.r, 'r_h') \
            .withColumnRenamed('n_assets', 'n_assets_h') \
            .withColumnRenamed(self.name, self.name + '_h') \
            .select(merge_vars + ['r_h', 'n_assets_h', self.name + '_h'])
        low = agged.filter(f.col('sig_rank') == 1) \
            .withColumnRenamed(self.r, 'r_l') \
            .withColumnRenamed('n_assets', 'n_assets_l') \
            .withColumnRenamed(self.name, self.name + '_l') \
            .select(merge_vars + ['r_l', 'n_assets_l', self.name + '_l'])
        high_minus_low = high.join(low, on=merge_vars) \
            .withColumn('sig_rank', f.lit('high_minus_low')) \
            .withColumn(self.r, f.col('r_h') - f.col('r_l')) \
            .withColumn('n_assets', f.col('n_assets_h') + f.col('n_assets_l')) \
            .drop('r_h', 'r_l', 'n_assets_h', 'n_assets_l')
        # combine high minus low with ranked data
        self.portr = agged \
            .sort('sig_rank') \
            .withColumn('sig_rank', f.col('sig_rank').cast(StringType())) \
            .unionByName(high_minus_low, allowMissingColumns=True) \
            .withColumn('trading_cost', 2 * f.col('n_assets') * turnover * commission / total_position) \
            .toPandas() \
            .sort_values(self.time) \
            .set_index(self.time)
        self.portr['port_r'] = self.portr.groupby(['sig_rank', 'cond_rank'])[self.r].shift(shift) \
            if 'cond_rank' in self.portr.columns else self.portr.groupby(['sig_rank'])[self.r].shift(shift)
        return self.portr

    def sort_return(self, days=1, average_across_2nd_ranks=False, trading_cost=False) -> PandasDataFrame:
        """
        data: data generated by method bivariate_portfolio_r()
        conditional portfolio
        Raw returns for equal???weighted portfolios sorted into ntiles.
        mean_return_net: mean return net of trading cost
        trade_cost: percentage trading cost.
        days: prediction horizaon. For example, if forward 1 day return, then days = 1
        """

        def describe(df):
            df = df[::days]
            r = (df['port_r'] - df['trading_cost']) if trading_cost is True else df['port_r']
            d = {}
            d['mean_return'] = r.mean()
            d['t_statistic'] = ttest_1samp(r, 0).statistic
            d['pvalue'] = ttest_1samp(r, 0).pvalue
            d['annual_SR'] = np.sqrt(252 / days) * r.mean() / r.std()
            d['annual_return'] = (1 + r.mean()) ** (252 / days) - 1
            d['n_assets'] = df['n_assets'].mean()
            for var in self.var_list:
                d[var + '_mean'] = df[var].mean()
            return pd.Series(d)

        returns = self.portr.loc[self.portr['port_r'].notnull()]
        if 'cond_rank' in self.portr.columns:
            if average_across_2nd_ranks is True:
                returns = returns.drop('cond_rank', axis=1).groupby([self.time, 'sig_rank']).mean()
                group_vars = ['sig_rank']
            else:
                group_vars = ['cond_rank', 'sig_rank']
        else:
            group_vars = ['sig_rank']
        return returns.groupby(group_vars).apply(describe)

    def get_transact_cost(self, single_trip_cost: float):
        r = self.portr.loc[self.portr['sig_rank'] == 'high_minus_low']
        r[self.name + '_h_lag'] = r[self.name + '_h'].shift(1)
        r[self.name + '_l_lag'] = r[self.name + '_l'].shift(1)
        for index, row in r.iterrows():
            try:
                turnover_h = len([i for i in row[self.name + '_h_lag'] if i not in row[self.name + '_h']]) \
                             / len(row[self.name + '_h_lag'])
                turnover_l = len([i for i in row[self.name + '_l_lag'] if i not in row[self.name + '_l']]) \
                             / len(row[self.name + '_l_lag'])
                turnover = (turnover_h + turnover_l) / 2
                r.at[index, 'turnover_h'], r.at[index, 'turnover_l'], r.at[index, 'turnover'] \
                    = turnover_h, turnover_l, turnover
            except Exception as e:
                print(e)
                r.at[index, 'turnover_h'], r.at[index, 'turnover_l'], r.at[index, 'turnover'] = np.nan, np.nan, np.nan
        r['port_r_after_fee'] = r['port_r'] - r['turnover'] * single_trip_cost * 2
        return r

    def sort_alpha(self, model='FF5', mom=True, freq='daily', average_across_2nd_ranks=False):
        """
        data: return of sorted portfolio returns per freq(daily, weekly, monthly, annually)
        model: 'FF3', 'FF5', 'q-factor'
        """

        def reg_OLS(pdf, factor_vars: List[str]):
            X = sm.add_constant(pdf[factor_vars])
            y = pdf['portr_excess']
            res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            res_df = pd.DataFrame({f"coeffs": res.params,
                                   f'tvalues': res.tvalues,
                                   f"pvals": res.pvalues
                                   }, index=X.columns)
            res_df.index.name = 'vars'
            return res_df

        # download factors
        if model == 'FF3' or model == 'FF5':
            factors = utils.get_FF_factors(model, mom, freq)
        elif model == 'q-factor':
            factors = utils.get_q_factors(mom, freq)
        # get factor variables
        factor_cols = [i for i in factors.columns if i != 'RF']
        if mom is False:
            factor_cols = [i for i in factor_cols if i != 'Mom']
        elif mom is True:
            factor_cols = factor_cols
        # get correct group variables and average portfolio return across different conditional variable if neeeed
        returns = self.portr.loc[self.portr['port_r'].notnull()]
        if 'cond_rank' in self.portr.columns:
            if average_across_2nd_ranks is True:
                returns = returns.drop('cond_rank', axis=1).groupby([self.time, 'sig_rank']).mean().reset_index(level=1)
                group_vars = ['sig_rank']
            else:
                group_vars = ['cond_rank', 'sig_rank']
        else:
            group_vars = ['sig_rank']
        # match returns with factors
        date_format = {'daily': '%Y%m%d', 'weekly': '%Y%m%d', 'monthly': '%Y%m', 'quarterly': '%Y%m', 'yearly': '%Y'}
        returns.index = returns.index.strftime(date_format[freq])
        df = returns.merge(factors, left_index=True, right_index=True, how='inner')
        df['portr_excess'] = df['port_r'] - df['RF']
        return df.groupby(group_vars).apply(reg_OLS, factor_vars=factor_cols)

    def sort_perf(self, r: PysparkDataFrame, FF_freq=None):
        def gen_perf(x):
            returns = x['port_r']
            excess_r = x['port_r'] - x['RF']

            perfs = dict()
            perfs['Average Return'] = returns.mean()
            perfs['Max Return'] = returns.max()
            perfs['Min Return'] = returns.min()
            perfs['Max Drawdown'] = np.abs(((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min())
            perfs['Volatility'] = returns.std()
            perfs['95%-VaR'] = returns.quantile(0.05)
            perfs['99%-VaR'] = returns.quantile(0.01)
            perfs['95%-ES'] = returns.loc[returns <= returns.quantile(0.05)].mean()
            perfs['99%-ES'] = returns.loc[returns <= returns.quantile(0.01)].mean()
            perfs['Sharpe Ratio'] = excess_r.mean() / excess_r.std()
            perfs['Naive Sharpe Ratio'] = returns.mean() / returns.std()
            perfs['Sortino Ratio'] = excess_r.mean() / excess_r.loc[excess_r < 0].std()
            perfs['Calmer Ratio'] = returns.mean() / perfs['Max Drawdown']
            perfs['Annualized Average Return'] = mult * perfs['Average Return']
            perfs['Annualized Geometric Return'] = (1 + returns).cumprod().values[-1] ** (mult / len(returns)) - 1
            perfs['Annual Volatility'] = np.sqrt(mult) * perfs['Volatility']
            perfs['Annual Sharpe Ratio'] = np.sqrt(mult) * perfs['Sharpe Ratio']
            perfs['Annual Sortino Ratio'] = np.sqrt(mult) * perfs['Sortino Ratio']
            perfs['Annual Calmer Ratio'] = mult * perfs['Calmer Ratio']
            perfs['Hit Rate'] = (returns > 0).sum() / len(returns)
            perfs['Loss Rate'] = (returns < 0).sum() / len(returns)
            perfs['Gross Profit'] = returns.loc[returns > 0].sum()
            perfs['Gross Loss'] = returns.loc[returns < 0].sum()
            perfs['Average Return in Up Days'] = returns.loc[returns > 0].mean()
            perfs['Average Return in Down Days'] = returns.loc[returns < 0].mean()
            perfs['Slugging Ratio'] = -1 * perfs['Average Return in Up Days'] / perfs['Average Return in Down Days']
            perfs['Profit Factor'] = -1 * perfs['Gross Profit'] / perfs['Gross Loss']
            perf_df = pd.DataFrame(perfs, index=['filler'])
            perf_df.index.name = 'filler'
            return perf_df

        mults = {'daily': 252, 'weekly': 52, 'monthly': 12, 'yearly': 1}
        mult = mults[FF_freq]

        rf = utils.get_FF_factors(FF_freq)[['date', 'RF']]
        date_format = {'daily': '%Y%m%d', 'weekly': '%Y%m%d', 'monthly': '%Y%m', 'yearly': '%Y'}
        pdf = utils.toPandas(r)
        pdf[self.time] = pdf[self.time].dt.strftime(date_format[FF_freq])
        pdf = pdf.merge(rf, left_on=self.time, right_on='date', how='inner')
        group_vars = ['weight_scheme', 'cond_rank', 'sig_rank'] if 'cond_rank' in pdf.columns \
            else ['weight_scheme', 'sig_rank']
        return pdf.groupby(group_vars) \
            .apply(gen_perf) \
            .droplevel('filler')
