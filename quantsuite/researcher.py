# class for empirical asset pricing papers.
from quantsuite.portfolio_analysis import PortfolioAnalysis
import os
from linearmodels import FamaMacBeth
import pandas as pd
from statsmodels.tsa.stattools import acf
from typing import List

class EAPResearcher:
    """
    Empirical asset pricing researcher
    """
    def __init__(self, data, time, stk_id, sig, forward_r):
        """
        data: pandas dataframe with time index
        """
        self.data_pdf = data
        self.time = time
        self.stk_id = stk_id
        self.sig = sig
        self.forward_r = forward_r


    @staticmethod
    def print2(msg, pad='-', total_len=50):
        if len(msg) >= total_len:
            print(msg)
        else:
            n1 = int((total_len - len(msg)) / 2)
            n2 = total_len - len(msg) - n1
            print(pad * n1 + msg + pad * n2)

    def get_avg_acf(self, nlags=5, acf_name='acf'):
        """
        calculate average autocorrelation function
        """
        acfs = self.data_pdf.loc[self.data_pdf[self.sig].notnull()]\
            .sort_values(self.time)\
            .groupby(self.stk_id)[self.sig]\
            .apply(lambda x: acf(x, nlags=nlags, missing='drop'))\
            .reset_index()
        acfs[[f'{acf_name}{i}' for i in range(nlags+1)]] = pd.DataFrame(acfs[self.sig].to_list())
        acfs = acfs.drop(self.sig, axis=1).mean()
        return acfs.loc[acfs.index != f'{acf_name}0']

    def summarize_sig(self, quantiles = [0.05, 0.25,  0.5, 0.75, 0.95], acf_nlags=5):
        summary = {}
        summary['unique_time'] = self.data_pdf[self.time].nunique()
        summary['unique_firm'] = self.data_pdf[self.stk_id].nunique()
        summary['n_observations'] = self.data_pdf[self.sig].count()

        summary['mean_cs'] = self.data_pdf.groupby(self.time)[self.sig].mean().mean()
        summary['mean_ts'] = self.data_pdf.groupby(self.stk_id)[self.sig].mean().mean()
        summary['mean_pool'] = self.data_pdf[self.sig].mean()

        summary['std_cs'] = self.data_pdf.groupby(self.time)[self.sig].std().mean()
        summary['std_ts'] = self.data_pdf.groupby(self.stk_id)[self.sig].std().mean()
        summary['std_pool'] = self.data_pdf[self.sig].std()

        quantiles = self.data_pdf[self.sig].quantile(quantiles)
        quantiles.index = ['quantile_' + str(i) for i in quantiles.index]

        self.data_pdf['pct_rank'] = self.data_pdf.groupby(self.time)[self.sig].rank(pct=True)

        summary = pd.Series(summary) \
            .append(quantiles)\
            .append(self.get_avg_acf(acf_nlags, 'raw_acf'))

        return summary

    def summarize_vars(self, vars: List[str], quantiles = [0.05, 0.25,  0.5, 0.75, 0.95]):
        summary1 = self.data_pdf[vars].describe().transpose()[['mean', 'std']]
        summary2 = self.data_pdf[vars].apply(lambda x: x.quantile(quantiles)).transpose()
        return pd.concat([summary1, summary2], axis = 1)

    def uni_sort(self, spark, shift: int, weight=None, covariates=None, ntile = 10, freq='daily', sort_alpha = True,
                 models = ['FF5', 'q-factor'],
                 mom=True):
        """
        Portfolio single sort, which provides sort on raw returns, coviariates, and alphss

        """
        # convert pandas to pyspark dataframe
        self.data_pdf.to_parquet('temp')
        data = spark.read.parquet('temp')
        # sorted raw return
        pa = PortfolioAnalysis(data, self.time, self.stk_id, self.sig, self.forward_r, weight, var_list=covariates)
        pa.gen_portr(shift=shift, ntile=ntile)
        sorted_returns = pa.sort_return().transpose()
        sorted_returns, sorted_vars = sorted_returns.iloc[0:2],sorted_returns.iloc[2:]
        # sorted alpha
        alphas_df = pd.DataFrame()
        if sort_alpha is True:
            for model in models:
                alphas = pa.sort_alpha(model, mom=mom, freq=freq).reset_index(level=1)
                alphas = alphas.loc[alphas['vars'] == 'const'][['coeffs', 'tvalues']]\
                    .rename(columns={'coeffs': model +'_alpha'})\
                    .transpose()
                alphas_df = alphas_df.append(alphas)
        # merged and rearrange columns
        sorted_results = pd.concat([sorted_returns, alphas_df, sorted_vars])
        rearraged_cols = [i for i in sorted_results.columns if i not in [str(ntile), 'high_minus_low']]\
                         + [str(ntile) , 'high_minus_low']
        return pa.portr, sorted_results[rearraged_cols]

    def double_sort(self, spark, shift: int, forward_r, models = ['FF5', 'q-factor'], mom=True, freq ='monthly',
                    sig_ntile=10, sort_var2=None, ntile2=None, dependent_sort = False, roundto=3):
        # convert pandas to pyspark dataframe
        self.data_pdf.to_parquet('temp')
        data = spark.read.parquet('temp')
        # double sort
        pa = PortfolioAnalysis(data, self.time, self.stk_id, self.sig, forward_r, var_list=['mktcap'])
        pa.gen_portr(shift=shift, ntile=sig_ntile, sort_var2=sort_var2, ntile2=ntile2, dependent_sort=dependent_sort)
        # double sorted return
        double_sort = {}
        sorted_r = pa.sort_return(average_across_2nd_ranks=False)[['mean_return', 't_statistic']]\
            .rename(columns ={'mean_return': 'value'})
        sorted_r['value'] = sorted_r['value']
        sorted_r_formatted = pd.DataFrame()
        for i in sorted_r.index.get_level_values('cond_rank').unique().tolist():
            sorted_r_one_rank = sorted_r.loc[sorted_r.index.get_level_values('cond_rank') == i].droplevel(0)\
                .transpose()\
                .reset_index()
            sorted_r_one_rank['cond_rank'] = i
            sorted_r_formatted = sorted_r_formatted.append(sorted_r_one_rank)
        double_sort['hedged_return'] = sorted_r_formatted
        # double sorted return
        for model in models:
            alpha = pa.sort_alpha(model=model, freq=freq, mom=mom, average_across_2nd_ranks=False)\
                [['coeffs', 'tvalues']]\
                .rename(columns = {'coeffs': 'value', 'tvalues': 't_statistic'})
            alpha['value'] = alpha['value']
            alpha = alpha.loc[alpha.index.get_level_values('vars') == 'const'].droplevel(2)
            alpha_formatted = pd.DataFrame()
            for i in alpha.index.get_level_values('cond_rank').unique().tolist():
                alpha_one_rank = alpha.loc[alpha.index.get_level_values('cond_rank') == i].droplevel(
                    0).transpose()\
                    .reset_index()
                alpha_one_rank['cond_rank'] = i
                alpha_formatted = alpha_formatted.append(alpha_one_rank)
            double_sort[f'{model}-alpha'] =  alpha_formatted
        # average returns across 2nd sorting variable
        sorted_r = pa.sort_return(average_across_2nd_ranks=True)[['mean_return', 't_statistic']]
        sorted_r['return_alpha'] = (sorted_r['mean_return']).round(roundto).astype(str) + \
                          '|' + sorted_r['t_statistic'].round(roundto).astype(str)

        high_minus_low_return = sorted_r[['return_alpha']].transpose()[['high_minus_low']] \
            .rename(columns={'high_minus_low': 'high_minus_low_return'})
        # average alphas across 2nd sorting variable
        alphas = []
        for model in models:
            alpha = pa.sort_alpha(model=model, freq=freq, mom=mom, average_across_2nd_ranks=True).reset_index(level=1)
            alpha = alpha.loc[alpha['vars'] == 'const']
            alpha['return_alpha'] = (alpha['coeffs']).round(roundto).astype(str) + \
                           '|' + alpha['tvalues'].round(roundto).astype(str)
            alpha = alpha[['return_alpha']].transpose()[['high_minus_low']].rename(columns={'high_minus_low': f'{model}_alpha'})
            alphas.append(alpha)

        averaged_return_alpha = pd.concat([high_minus_low_return] + alphas, axis=1)
        return {'double_sorted': double_sort, 'double_sorted_average': averaged_return_alpha}

    def FM_regression(self, covariates):

        df_FM = self.data_pdf.set_index([self.stk_id, self.time])
        y = df_FM[self.forward_r]
        X  = df_FM[[self.sig] + covariates]
        X['constant']= 1
        res = FamaMacBeth(y, X).fit(cov_type='kernel', kernel='bartlett',  bandwidth=5)
        # format output table
        table_v= pd.DataFrame([res.params, res.tstats]).transpose()
        # table_v['parameter'] = table_v['parameter']
        # table_v['star'] = ''
        # table_v.loc[table_v['pvalue'] <= 0.01, 'star'] = "***"
        # table_v.loc[(0.01 < table_v['pvalue']) & (table_v['pvalue'] <= 0.05), 'star'] = "**"
        # table_v.loc[(0.05 < table_v['pvalue']) & (table_v['pvalue'] <= 0.1), 'star'] = "*"
        # table_v['parameter'] = table_v['parameter'].round(round_to).astype(str) + table_v['star'] + '|'+\
        #                         table_v['tstat'].round(round_to).astype(str)
        #
        # table_v = table_v[['parameter']]
        # table_h = table_v.transpose()
        return table_v

    def double_sort_FM(self, covariates, sort_var, sort_n):
        # generate rank--------------------------------------
        data = self.data_pdf.loc[self.data_pdf[sort_var].notnull()]
        data['rank2'] = data.groupby(self.time)[sort_var]\
            .apply(lambda x: pd.qcut(x, sort_n, labels=range(1, sort_n + 1)))
        results = []
        for i in range(sort_n):
            eapr = EAPResearcher(data.loc[data['rank2'] == i + 1], self.time, self.stk_id, self.sig, self.forward_r)
            res = eapr.FM_regression(covariates=covariates)
            res = res.loc[res.index.values == self.sig]
            res['rank'] = i + 1
            results.append(res)
        return pd.concat(results)
