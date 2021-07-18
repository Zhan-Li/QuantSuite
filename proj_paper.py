# evaluate signal using stocks as investment
# bas performance with 0.61 SR in the 15-20 test period
# generally, when aggregating option level data to stock level data, value to be aggregated should be positive for call
# and nagetive for put, and then weight need to be positive.
import pandas as pd
import pyspark.sql.functions as f
from pyspark import StorageLevel
import importlib
import sig_option
import utilities

importlib.reload(sig_option)
from sig_option import OptionSignal
from pyspark.sql.types import  StringType
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
import sig_evaluator; importlib.reload(sig_evaluator)
from sig_evaluator import PortfolioAnalysis
import utilities as utils
import quantstats
import sig_technical; importlib.reload(sig_technical)
import numpy as np
from sig_technical import TechnicalSignal
import optuna
import joblib
import data_manipulator
importlib.reload(data_manipulator)
from data_manipulator import DataManipulator
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import datetime
from statsmodels.tsa.stattools import acf
from linearmodels import FamaMacBeth
import dataframe_image as dfi


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
        .set("spark.executor.memory", '50g') \
        .set("spark.driver.memory", "50g") \
        .set('spark.driver.maxResultSize', '2g')\
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

#people.write.format("mongo").mode("append").save()
# fixed parameters
start_date = '2011-01-01'
end_date = '2020-12-31'
forward_r = 'forward_r'
vold_min = 200000
sig_name = 'ks|opt_vold_CP'
MILLION = 1000000
BILLION = 1000000000



# security  data----------------------------------------
sec = spark.read.parquet('data/dsf_linked')
# y
sec = DataManipulator(sec, 'date', 'cusip')\
    .add_cumr('ret', 1, 1, 'forward_r')\
    .add_cumr('ret', 2, 2, 'forward_r2')\
    .add_cumr('ret', 3, 3, 'forward_r3')\
    .add_cumr('ret', 4, 4, 'forward_r4')\
    .add_cumr('ret', 5, 5, 'forward_r5') \
    .add_cumr('ret', 1, 2, 'forward_r12') \
    .add_cumr('ret', 1, 3, 'forward_r13') \
    .add_cumr('ret', 1, 4, 'forward_r14') \
    .add_cumr('ret', 1, 5, 'forward_r15') .data
# X----------------------------------------
## sec vars
sec = sec.withColumn('mktcap', f.col('mktcap')/BILLION)\
    .withColumn('turnover', f.col('vol')/f.col('shrout'))\
    .withColumn('illiq', f.abs('ret')/f.col('vol_d')*MILLION)\
    .withColumn('illiq', f.mean('illiq').over(Window.partitionBy('cusip').orderBy('date').rowsBetween(-20, 0)))
sec = TechnicalSignal(sec, 'date', 'cusip')\
        .add_std('ret', -20, 0, 'ret_std')\
        .add_past_cumr('ret', -20, 0, 'cum_r1') \
        .add_past_cumr('ret', -252, -20, 'cum_r2').data
## fundamental vars
fundq_vars = spark.read.parquet('data/fundq')\
    .filter((f.col('date') >= start_date) & (f.col('date') <= end_date))
## option vars
opt = spark.read.parquet('data/opprcd_1996_2020') \
    .filter((f.col('date') >= start_date) & (f.col('date') <= end_date))\
    .join(sec, on=['date', 'cusip'])\
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('ks', f.col('strike_price')/f.col('prc')) \
    .withColumn('ks>1?', f.when(f.col('ks') >1, True).when(f.col('ks') <= 1, False))\
    .withColumn('ks+-', f.when(f.col('cp_flag') == 'P', -1*f.col('ks')).otherwise(f.col('ks'))) \
    .withColumn('IV+-', f.when(f.col('cp_flag') == 'P', -1 * f.col('impl_volatility')).otherwise(f.col('impl_volatility'))) \
    .withColumn('exdays+-',f.when(f.col('cp_flag') == 'P', -1*f.col('exdays')).otherwise(f.col('exdays')))\
    .withColumn('elasticity', f.abs('delta') * f.col('prc') / f.col('opt_prc'))\
    .withColumn('elasticity+-',f.when(f.col('cp_flag') == 'P', -1*f.col('elasticity')).otherwise(f.col('elasticity')))\
    .withColumn('opt_vol_adj+-', f.when(f.col('cp_flag') == 'P', -1*f.col('opt_vol_adj')).otherwise(f.col('opt_vol_adj')))\
    .withColumn('opt_vold+-', f.when(f.col('cp_flag') == 'P', -1*f.col('opt_vold')).otherwise(f.col('opt_vold'))) \
    .withColumn('open_interest_adj+-',f.when(f.col('cp_flag') == 'P', -1 * f.col('open_interest_adj')).otherwise(f.col('open_interest_adj')))\
    .withColumn('oi_adj_change+-', f.when(f.col('cp_flag') == 'P', -1*f.col('oi_adj_change')).otherwise(f.col('oi_adj_change')))

weights = [None, 'opt_vold']
OTM_P = (f.col('cp_flag') == 'P') & ((f.col('strike_price') <= 0.95*f.col('prc')) | (f.col('strike_price') >= 0.8*f.col('prc')))
ATM_C = (f.col('cp_flag') == 'C') & ((f.col('strike_price') >= 0.95*f.col('prc')) | (f.col('strike_price') <= 1.05*f.col('prc')))
opt_sigs = []
for weight in weights:
    IV_skew_sig =  OptionSignal(opt.filter(OTM_P|ATM_C), 'date', 'cusip', 'cp_flag', 'C', 'P')\
        .gen_IV_skew('IV+-', weight, 'ks>1?', f'IV_skew|{weight}')\
        .select('date', 'cusip', f.col(f'IV_skew|{weight}_P_plus_C'))\
        .withColumn(f'IV_skew|{weight}_P_plus_C', -f.col(f'IV_skew|{weight}_P_plus_C'))
    OS = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P')
    IV_spread_sig = OS.gen_IV_spread('IV+-', weight, f'IV_spread|{weight}')
    VRP = OS.gen_rv_iv_spread('ret_std', 'impl_volatility', weight, sec, f'rv_iv_spread|{weight}')
    lvg_sig1 = OS.gen_ks('ks+-', weight, f'ks|{weight}').drop(f'ks|{weight}_C', f'ks|{weight}_P')
    lvg_sig2 = OS.gen_expdays('exdays+-', weight, f'expdays|{weight}')
    lvg_sig3 = OS.gen_elasticity('elasticity+-', weight, f'elasticity|{weight}').drop(f'elasticity|{weight}_C', f'elasticity|{weight}_P')
    IV_avg = OS.gen_aggregate_sig('impl_volatility', weight, f'IV_avg|{weight}')
    opt_sigs = opt_sigs + [IV_skew_sig, IV_spread_sig, VRP, lvg_sig1, lvg_sig2, lvg_sig3, IV_avg]

opt_vars = utils.multi_join(opt_sigs, on=['date', 'cusip']).persist(storageLevel=StorageLevel.DISK_ONLY)

# merge stock return with option signal
merged = sec \
    .filter((f.col('date') >= start_date) & (f.col('date') <= end_date)) \
    .filter((f.col('shrcd') == 10) | (f.col('shrcd') == 11)) \
    .filter(f.col('prc') > 5) \
    .join(fundq_vars, on=['date', 'cusip'])\
    .join(opt_vars, on=['date', 'cusip'])\
    .select('date', 'cusip', *[i for i in sec.columns if 'forward_r' in i], 'ret', 'turnover', 'illiq', 'prc', 'vol_d', 'mktcap', 'ret_std', 'cum_r1', 'cum_r2',
            'BM',
            *[i for i in opt_vars.columns if i not in ['date', 'cusip'] and 'None' not in i])\
    .persist(storageLevel=StorageLevel.DISK_ONLY)
merged.write.mode('overwrite').parquet('merged')

# down sampling
merged = spark.read.parquet('merged').cache()
df_daily = merged.toPandas()

# resample dataset if needed. Unfinished.
def resample_r(df, id, r, resample_rule, resampled_r='resampled_r'):
    resampled_df = df \
        .assign(log_gross_r=np.log(1 + df[r]))[[id, 'log_gross_r']] \
        .groupby(id).resample(resample_rule).sum()
    resampled_df[resampled_r] = np.exp(resampled_df['log_gross_r']) - 1
    return resampled_df[[resampled_r]]

def resample(df, id, resample_rule, agg_rule='last'):
    resampled = df.groupby(id).resample(resample_rule)
    if agg_rule == 'last':
        return resampled.last()
    elif agg_rule == 'mean':
        return resampled.mean()

"""
r_weekly = resample_r(df_daily.set_index('date'), 'cusip', 'ret', 'W-Tue', 'ret_w')
r_weekly['forward_r_w'] = r_weekly.groupby('cusip')['ret_w'].shift(-1)

r_monthly = resample_r(df_daily.set_index('date'), 'cusip', 'ret', 'M', 'ret_m')
r_monthly['forward_r'] = r_monthly.groupby('cusip')['ret_m'].shift(-1)
other_monthly = resample(df_daily.set_index('date'), 'cusip', 'M', 'mean').drop(['forward_r'], axis = 1)
df_monthly = r_monthly.join(other_monthly)
"""

class EAPResearcher:
    """
    Empirical asset pricing researcher
    """
    def __init__(self, data, time, stk_id, sig):
        """
        data: pandas dataframe with time index
        """
        self.data_df = data
        self.stk_id = stk_id
        self.sig = sig
        self.time = time

    @staticmethod
    def print2(msg, pad='-', total_len=50):
        if len(msg) >= total_len:
            print(msg)
        else:
            n1 = int((total_len - len(msg)) / 2)
            n2 = total_len - len(msg) - n1
            print(pad * n1 + msg + pad * n2)

    def check_data_pdf(self):
        if hasattr(self, 'data_pdf') is False:
            print('Converting PySpark Dataframe to Pandas Dataframe...')
            self.data_pdf = self.data_df.toPandas()
            print('Conversion finished.')


    def get_avg_acf(self, nlags=5, acf_name='acf'):
        """
        calculate average autocorrelation function
        """
        self.check_data_pdf()
        acfs = self.data_pdf.loc[self.data_pdf[self.sig].notnull()]\
            .sort_values(self.time)\
            .groupby(self.stk_id)[self.sig]\
            .apply(lambda x: acf(x, nlags=nlags, missing='drop'))\
            .reset_index()
        acfs[[f'{acf_name}{i}' for i in range(nlags+1)]] = pd.DataFrame(acfs[self.sig].to_list())
        acfs = acfs.drop(self.sig, axis=1).mean()
        return acfs.loc[acfs.index != f'{acf_name}0']

    def summarize(self, acf_nlags=5, q=[0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]):
        self.check_data_pdf()
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

        quantiles = self.data_pdf[self.sig].quantile(q)
        quantiles.index = ['quantile_' + str(i) for i in quantiles.index]

        self.data_pdf['pct_rank'] = self.data_pdf.groupby(self.time)[self.sig].rank(pct=True)

        summary = pd.Series(summary) \
            .append(quantiles)\
            .append(self.get_avg_acf(acf_nlags, 'raw_acf'))

        return summary

    def sort(self, forward_r, weight, covariates):
        """
        Portfolio single sort, which provides sort on raw returns, coviariates, and alphss

        """
        pa = PortfolioAnalysis(self.data_df, self.time, self.stk_id, self.sig, forward_r, weight, var_list=covariates)
        pa.gen_portr(ntile=10)
        sorted_results = pa.sort_return()

        if covariates is not None:
            covariates_mean = \
            sorted_results[[i + '_mean' for i in covariates] + [sig_name +'_mean']].transpose().drop('high_minus_low', axis=1) \
                [[str(i) for i in range(1, 11)]]
        else:
            covariates_mean = np.nan

        ## sorted raw_return
        raw_returns = sorted_results[['mean_return', 't_statistic']].transpose()
        raw_returns['model'] = 'raw_return'
        raw_returns = raw_returns[['model'] + [str(i) for i in range(1, 11)] + ['high_minus_low']]
        ## sorted alpha
        alphas_df = pd.DataFrame()
        for model in ['FF5', 'q-factor']:
            alphas = pa.sort_alpha(model, mom=True, freq='daily')
            alphas = alphas.xs('const', level=1, drop_level=True).drop('pvals', axis=1).transpose()
            alphas['model'] = f'{model}_mom'
            alphas = alphas[['model'] + [str(i) for i in range(1, 11)] + ['high_minus_low']]
            alphas_df = alphas_df.append(alphas)
        ## combine

        return raw_returns.append(alphas_df), covariates_mean

    def FM_regression(self, forward_r, covariates, vertical_table = True):

        self.check_data_pdf()
        df_FM = self.data_pdf.set_index([self.stk_id, self.time])
        y = df_FM[forward_r]
        X  = df_FM[[self.sig] + covariates]
        res = FamaMacBeth(y, X).fit(cov_type='kernel', kernel='bartlett',  bandwidth=3)
        # format output table
        table_v= pd.DataFrame([res.params, res.tstats, res.pvalues]).transpose()
        table_v['parameter'] = table_v['parameter'].astype(str)
        table_v['star'] = ''
        table_v.loc[x['pvalue'] <= 0.01, 'star'] = "***"
        table_v.loc[(0.01 < table_v['pvalue']) & (table_v['pvalue'] <= 0.05), 'star'] = "**"
        table_v.loc[(0.05 < table_v['pvalue']) & (table_v['pvalue'] <= 0.1), 'star'] = "*"
        table_v['parameter'] = table_v['parameter'] + table_v['star']
        table_v = table_v[['parameter', 'tstat']]
        table_h = table_v.transpose()
        table_v = table_v.reset_index().rename(columns={'index': 'vars'})
        table_v = pd.melt(table_v, id_vars=['vars'], value_vars=['parameter', 'tstat'], var_name='result')\
            .sort_values(['vars', 'result'])
        return table_v if vertical_table is True else table_h





# summarization
df = merged
dfs = {'full': df, '2011-2015':df.filter(f.col('date')<='2015-12-31'),  '2016-2020': df.filter(f.col('date')>'2015-12-31')}
summary_list = []
for sig_name in ['ks|opt_vold_CP', 'expdays|opt_vold_CP']:
    summaries = pd.DataFrame()
    for key, value in dfs.items():
        eapr = EAPResearcher(value, 'date', 'cusip', sig_name)
        summary = eapr.summarize()
        summaries[key] = summary
    summaries['sig'] =  sig_name
    summary_list.append(summaries)
summary_df = pd.concat(summary_list)
dfi.export(summary_df,"summaries.png")

# portfolio analyais
covariates = ['mktcap', 'BM', 'ret_std','turnover', 'illiq', 'cum_r1', 'cum_r2',
                     'IV_skew|opt_vold_P_plus_C', 'IV_spread|opt_vold',  'rv_iv_spread|opt_vold']
for sig_name in ['ks|opt_vold_CP', 'expdays|opt_vold_CP']:
    eapr = EAPResearcher(merged, 'date', 'cusip', sig_name)
    alpha, vars = eapr.sort(forward_r = forward_r,
                            weight = None,
                            covariates = ['expdays|opt_vold_CP'] + covariates if sig_name == 'ks|opt_vold_CP' else ['ks|opt_vold_CP'] + covariates)
    dfi.export(alpha,f"alpha_{sig_name}.png")
    dfi.export(vars, "vars.png")




# # how long does predictability last?
# alpha_list = []
# for r in [i for i in merged.columns if 'forward_r' in i]: # different forward_r represents different forcast horizon
#     alpha, _ = sort(data=merged, time='date', id='cusip', sig=sig_name, forward_r=r, weight=None,
#                        covariates=None)
#     alpha['r'] = r
#     alpha_list.append(alpha)
# alphas = pd.concat(alpha_list)
# alphas.sort_values(['model','r'])




# Fama–MacBeth Regressions
covariates_FM = ['expdays|opt_vold_CP', 'turnover', 'illiq', 'mktcap', 'ret_std', 'cum_r1', 'cum_r2', 'BM',
                 'IV_skew|opt_vold_P_plus_C', 'IV_spread|opt_vold',  'rv_iv_spread|opt_vold']
eapr = EAPResearcher(merged, 'date', 'cusip', 'ks|opt_vold_CP')
res1 = eapr.FM_regression(forward_r = forward_r, covariates = covariates_FM)





##  Fama–MacBeth Regressions add interaction
merged2 = merged.withColumn('KS_interaction', f.col('expdays|opt_vold_CP')*f.col('ks|opt_vold_CP'))
covariates_FM2 = ['expdays|opt_vold_CP', 'turnover', 'illiq', 'mktcap', 'ret_std', 'cum_r1', 'cum_r2', 'BM',
                 'IV_skew|opt_vold_P_plus_C', 'IV_spread|opt_vold',  'rv_iv_spread|opt_vold', 'KS_interaction']
eapr = EAPResearcher(merged2, 'date', 'cusip', 'ks|opt_vold_CP')
res2 = eapr.FM_regression(forward_r = forward_r, covariates = covariates_FM2)

# What explain the predictability
## earnings surprise and CAR
## relationship with respect to mispricing and informted measures such PIN measure, signed put-call ratio.
## interaction effecg
### predictability is different with different expdays

