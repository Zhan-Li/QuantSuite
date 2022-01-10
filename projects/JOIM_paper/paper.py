import matplotlib
import pyspark.sql.functions as f
import seaborn as sns
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Window

import quantsuite.misc_funcs as utils
from quantsuite import EAPResearcher
from quantsuite import PerformanceEvaluation, PortfolioAnalysis
from quantsuite.signals import TechnicalSignal, OptionSignal

matplotlib.rcParams['text.usetex'] = False
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# spark
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
spark_conf = SparkConf() \
    .set("spark.executor.memory", '60g') \
    .set("spark.driver.memory", "60g") \
    .set('spark.driver.maxResultSize', '2g') \
    .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
    .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
# fixed parameters
MILLION = 1000000
BILLION = 1000000000
ROUNDTO = 3
START = '2005-01-01'
END = '2020-12-31'
# security  data----------------------------------------
sec = spark.read.parquet('data/dsf_linked') \
    .filter((f.col('date') >= '1996-01-01') & (f.col('date') <= '2020-12-31')) \
    .filter((f.col('shrcd') == 10) | (f.col('shrcd') == 11))
sec = TechnicalSignal(sec, 'date', 'cusip') \
    .add_std('ret', -20, 0, 'ret_std').data
# X----------------------------------------
## sec vars
sec_vars = sec \
    .withColumn('mktcap', f.col('mktcap') / BILLION) \
    .withColumn('turnover', f.col('vol') / f.col('shrout')) \
    .withColumn('illiq', f.abs('ret') / f.col('vol_d') * MILLION) \
    .withColumn('illiq', f.mean('illiq').over(Window.partitionBy('cusip').orderBy('date').rowsBetween(-20, 0))) \
    .select('date', 'cusip', 'turnover', 'illiq', 'prc', 'vol_d', 'mktcap')

sec_vars2 = TechnicalSignal(sec, 'date', 'cusip') \
    .add_cumr('ret', -20, 0, 'cum_r1') \
    .add_cumr('ret', -251, -21, 'cum_r2').data \
    .select('date', 'cusip', 'ret_std', 'cum_r1', 'cum_r2')
## fundamental vars
fundq_vars = spark.read.parquet('data/fundq') \
    .filter((f.col('date') >= '1996-01-01') & (f.col('date') <= '2020-12-31')) \
    .select('date', 'cusip', 'BM')
sec_vars.join(fundq_vars, on=['date', 'cusip']).describe().show()
## option vars
opt = spark.read.parquet('data/opprcd_1996_2020') \
    .withColumn('cp_flag', f.when(f.col('cp_flag') == 'c', 'C').otherwise(f.col('cp_flag'))) \
    .filter((f.col('date') >= '1996-01-01') & (f.col('date') <= '2020-12-31')) \
    .join(sec, on=['date', 'cusip']) \
    .withColumn('rv-iv', f.col('ret_std') * np.sqrt(252 / 20) - f.col('impl_volatility')) \
    .withColumn('opt_prc', (f.col('best_bid') + f.col('best_offer')) / 2) \
    .withColumn('ks', f.col('strike_price') / f.col('prc')) \
    .withColumn('elasticity', f.abs('delta') * f.col('prc') / f.col('opt_prc'))
OTM_P = (f.col('cp_flag') == 'P') & (
            (f.col('strike_price') <= 0.95 * f.col('prc')) | (f.col('strike_price') >= 0.8 * f.col('prc')))
ATM_C = (f.col('cp_flag') == 'C') & (
            (f.col('strike_price') >= 0.95 * f.col('prc')) | (f.col('strike_price') <= 1.05 * f.col('prc')))
weight = 'opt_vold'
IV_skew_sig = OptionSignal(opt.filter(OTM_P | ATM_C), 'date', 'cusip', 'cp_flag', 'C', 'P', weight) \
    .gen_IV_skew('impl_volatility', 'ks', f'IV_skew|{weight}') \
    .select('date', 'cusip', f'IV_skew|{weight}_P_minus_C')
OS = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight)
IV_sig = OS.gen_IV('impl_volatility', f'IV|{weight}').select('date', 'cusip', f'IV|{weight}_avg',
                                                             f'IV|{weight}_spread1')
VRP = OS.gen_rv_iv_spread('rv-iv', f'rv_iv_spread|{weight}').select('date', 'cusip', f'rv_iv_spread|{weight}_avg')
ks_sigs = []
weights = ['opt_vold', 'opt_vol_adj', 'open_interest_adj', None]
for weight in weights:
    OS = OptionSignal(opt, 'date', 'cusip', 'cp_flag', 'C', 'P', weight)
    ks = OS.gen_ks('ks', f'ks|{weight}').select('date', 'cusip', f'ks|{weight}_spread2')
    ks_sigs.append(ks)
opt_sigs = [IV_skew_sig, IV_sig, VRP] + ks_sigs
opt_vars = utils.multi_join(opt_sigs, on=['date', 'cusip'])

## all vars
vars_daily = sec_vars \
    .join(sec_vars2, on=['date', 'cusip']) \
    .join(fundq_vars, on=['date', 'cusip']) \
    .join(opt_vars, on=['date', 'cusip'])
# y-------------------------------------
sec_ret = TechnicalSignal(sec, 'date', 'cusip') \
    .add_cumr('ret', 1, 1, 'forward_r').data \
    .select('date', 'cusip', 'ret', 'forward_r')
# merge stock return with option signal
sec_ret.join(vars_daily, on=['date', 'cusip']).write.mode('overwrite').parquet(f'merged_daily_1996-01-01')

# read and resample data-------------------------------------
SIGS = ['ks|opt_vold_spread2', 'ks|opt_vol_adj_spread2', 'ks|open_interest_adj_spread2', 'ks|None_spread2']
FREQ = 'daily'
FORWORDR = 'forward_r'
RESAMPLED_R = 'ret'
MOM_FACTOR = False
SIGS = ['ks|opt_vold_spread2', 'ks|open_interest_adj_spread2']
df_daily = pd.read_parquet(f'merged_daily_1996-01-01')
df_daily = df_daily.loc[(df_daily['date'] >= START) & ((df_daily['date'] <= END))]
df_daily[FORWORDR] = df_daily[FORWORDR] * 100
# df_monthly = utils.downsample_from_daily(df_daily, 'date', 'cusip', r='ret', vars=df_daily.iloc[:, 4:].columns.to_list(), to_freq='monthly',
# var_agg_rule='mean',
# resampled_r=RESAMPLED_R, resampled_forward_r=FORWORDR)

df = df_daily

# summarization--------------------------------------------------------------------------
## summarize sig
dfs = {'full': df,
       '2005-2009': df.loc[df['date'] <= '2009-12-31'],
       '2010-2015': df.loc[(df['date'] >= '2010-01-01') & (df['date'] <= '2015-12-31')],
       '2016-2020': df.loc[df['date'] >= '2016-01-01']}
for SIG in SIGS:
    summary_list = []
    summaries = pd.DataFrame()
    for key, value in dfs.items():
        eapr = EAPResearcher(value, 'date', 'cusip', SIG, FORWORDR)
        summary = eapr.summarize_sig(acf_nlags=10)
        summaries[key] = summary
    summary_list.append(summaries)
    summary_df = pd.concat(summary_list)
    print(f'Result for {SIG}\n', summary_df)
    with open(f'summary_sig_{SIG}.tex', 'w') as tex:
        tex.write(summary_df.round(2).to_latex())
# summarize vars
eapr = EAPResearcher(df, 'date', 'cusip', SIG, FORWORDR)
summary_vars = eapr.summarize_vars([SIG, 'mktcap', 'BM', 'turnover', 'illiq'])
print(summary_vars)
with open('summary_vars.tex', 'w') as tex:
    tex.write(summary_vars.round(4).to_latex())
df[SIGS].corr()
# portfolio analyais ---------------------------------------------------------------
covariates = ['mktcap', 'BM', 'ret_std', 'turnover', 'illiq', 'cum_r1', 'cum_r2',
              'IV_skew|opt_vold_P_minus_C', 'IV|opt_vold_spread1', 'rv_iv_spread|opt_vold_avg']

for sig_name in SIGS:
    eapr = EAPResearcher(df, 'date', 'cusip', sig_name, forward_r=FORWORDR)
    port_r, sorted_res = eapr.uni_sort(spark, shift=1, ntile=5,
                                       weight='None',
                                       covariates=covariates, freq=FREQ, mom=True, models=['FF5', 'q-factor'])
    port_r = port_r.loc[port_r['sig_rank'] == 'high_minus_low'][['port_r']].reset_index()
    plt.clf()
    sns.lineplot(x='date', y='port_r', data=port_r)
    plt.xlabel("Year", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("%", fontsize=15)
    plt.savefig(f'long_short_{sig_name}.png')

    for row in [0, 2, 4]:
        sorted_res.iloc[row, :] = sorted_res.iloc[row, :]
    with open(f'sort_alpha_{sig_name}.tex', 'w') as tex:
        tex.write(sorted_res.round(3).to_latex())
    print(f'alphas for {sig_name} are:\n', sorted_res)
# Fama–MacBeth Regressions-----------------------------------------------------------------
covariates_FM1 = []
covariates_FM2 = ['turnover', 'illiq', 'mktcap', 'ret_std', 'cum_r1', 'cum_r2', 'BM']
covariates_FM3 = ['turnover', 'illiq', 'mktcap', 'ret_std', 'cum_r1', 'cum_r2', 'BM',
                  'IV_skew|opt_vold_P_minus_C', 'IV|opt_vold_spread1', 'rv_iv_spread|opt_vold_avg']
for SIG in SIGS:
    eapr = EAPResearcher(df, 'date', 'cusip', SIG, forward_r=FORWORDR)
    res1 = eapr.FM_regression(covariates=covariates_FM1)
    res2 = eapr.FM_regression(covariates=covariates_FM2)
    res3 = eapr.FM_regression(covariates=covariates_FM3)
    FMs = pd.concat([res1, res2, res3], axis=1)
    FMs.round(3).to_latex(f'FM_{SIG}.tex')
    print(f'FM regression results for {SIG} are:\n', FMs)

# double sort----------------------------------------------------------------------------------------
eapr = EAPResearcher(df, 'date', 'cusip', 'ks|opt_vold_spread2', FORWORDR)
for var in ['mktcap', 'prc', 'illiq', 'BM', 'cum_r1', 'cum_r2']:
    double_sort = eapr.double_sort(spark, shift=1, freq=FREQ, forward_r=FORWORDR, sig_ntile=10, ntile2=3, sort_var2=var,
                                   dependent_sort=True, mom=True, models=['FF5', 'q-factor'])
    sort_list = []
    for i in ['hedged_return', 'FF5-alpha', 'q-factor-alpha']:
        double_sort1 = double_sort['double_sorted'][i][['cond_rank', 'index', 'high_minus_low']] \
            .set_index(['cond_rank', 'index']) \
            .rename(columns={'high_minus_low': i})
        sort_list.append(double_sort1)
    double_sort_df = pd.concat(sort_list, axis=1)
    print(f'sorted by {var}', double_sort_df)
    double_sort_df.round(3).to_latex(f'double_sort_{var}.tex')

eapr.double_sort(spark, freq=FREQ, forward_r=FORWORDR, sig_ntile=5, ntile2=5, sort_var2='mktcap', dependent_sort=True)
# subgroup FM regression ----------------------------------------------------------------------------------------
for SIG in SIGS:
    n = 3
    eapr = EAPResearcher(df, 'date', 'cusip', SIG, forward_r=FORWORDR)
    results = []
    for var in ['mktcap', 'prc', 'illiq', 'BM', 'cum_r1', 'cum_r2']:
        res = eapr.double_sort_FM(covariates_FM3, sort_var=var, sort_n=3)
        res['var'] = var
        results.append(res)
    pd.concat(results, axis=1).round(2).to_latex(f'FM_subgroup_{SIG}.tex')
    print(f'subgroup FM for {SIG} is\n', pd.concat(results, axis=1))
# how long does predictability last?----------------------------------------------------------------
acf_list = []
for SIG in SIGS:
    alpha_list = []
    coef_list = []
    eapr = EAPResearcher(df, 'date', 'cusip', SIG, forward_r=f'forward_r')
    acfs = eapr.get_avg_acf(15, 'acf')
    print(f'acf for {SIG} is \n ', acfs)
    acf_list.append(acfs)
    pd.DataFrame(acf_list).transpose().round(3).to_latex(f'acf.tex')
    for i in [1, 2, 3, 4, 5, 10, 15]:
        df[f'forward_r{i}'] = df.sort_values('date').groupby('cusip')['ret'].shift(-i)
        eapr = EAPResearcher(df, 'date', 'cusip', SIG, forward_r=f'forward_r{i}')
        _, sorted_res = eapr.uni_sort(spark, ntile=5, shift=i, weight='None', covariates=None, freq=FREQ)
        sorted_res.columns.name = ''
        sorted_res = sorted_res[['high_minus_low']].iloc[0:6, :].rename(columns={'high_minus_low': 'day_' + str(i)})
        res = eapr.FM_regression(covariates=covariates_FM3)
        res = res.loc[res.index == SIG].rename(columns={'parameter': 'day_' + str(i)})
        coef_list.append(res)
        alpha_list.append(sorted_res)
    alphas = pd.concat(alpha_list, axis=1)
    alphas.iloc[0:6:2] = alphas.iloc[0:6:2].apply(lambda x: x * 100)
    alphas.round(3).to_latex(f'long_horizon_{SIG}.tex')
    coefs = pd.concat(coef_list, axis=1)
    print(f'Hedged returns and alphsa for {SIG} over long horizon is:\n', alphas)
    print(f'For {SIG}, The coefficient before the signal over long horizon is:\n', coefs)
# time series performance -----------------------------\
TRANSACTION_COST = True
SPY_daily = utils.download_stk_return('2005-01-01', '2020-12-31', 'SPY')[['Date', 'r']] \
    .assign(name='SPY').dropna()
perfs = []
extremes = []

for freq in ['daily', 'quarterly']:
    SPY_freq = \
        utils.downsample_from_daily(data=SPY_daily, time='Date', name='name', r='r', vars=[], to_freq=freq,
                                    var_agg_rule='last', resampled_r='r', resampled_forward_r='forward_SPY') \
            .set_index('Date')['r'] if freq != 'daily' else SPY_daily.set_index('Date')['r']
    rs = []
    turnovers = []
    for SIG in SIGS:
        # generate portfolio return
        df_freq = utils.downsample_from_daily(df_daily, 'date', 'cusip', r='ret', vars=[SIG],
                                              to_freq=freq, var_agg_rule='mean',
                                              resampled_r=RESAMPLED_R,
                                              resampled_forward_r=FORWORDR) if freq != 'daily' else df_daily
        df_freq.to_parquet('df_freq')
        pa = PortfolioAnalysis(spark.read.parquet('df_freq'), 'date', 'cusip', SIG, FORWORDR)
        port_r = pa.gen_portr(shift=1, ntile=10)
        r = port_r.loc[port_r['sig_rank'] == 'high_minus_low'].rename(columns={'port_r': f'port_r_{SIG}'})[
            f'port_r_{SIG}']
        if TRANSACTION_COST is True:
            transaction_cost = pa.get_transact_cost(8.37 / 10000)
            turnover = transaction_cost.rename(columns={'turnover': f'turnover_{SIG}'})[f'turnover_{SIG}']
            turnovers.append(turnover)
            r = transaction_cost \
                .rename(columns={'port_r_after_fee': f'port_r_after_fee_{SIG}'})[f'port_r_after_fee_{SIG}']

        rs.append(r)

        pa = PerformanceEvaluation(r)
        stats = pa.get_stats(freq=freq)
        stats['sig'] = SIG
        stats['freq'] = freq
        perfs.append(stats)

        pa = PerformanceEvaluation(SPY_freq.dropna())
        SPY_stats = pa.get_stats(freq=freq)
        SPY_stats['sig'] = 'SPY'
        SPY_stats['freq'] = freq
        perfs.append(SPY_stats)

        extreme = pd.DataFrame([r, SPY_freq]).transpose().dropna().sort_values('r').head(10).reset_index()
        extreme['sig'] = SIG
        extreme['freq'] = freq
        extremes.append(extreme)

    # plot turnover
    if TRANSACTION_COST is True:
        turnover_df = pd.concat(turnovers, axis=1)
        summary1 = turnover_df.describe().transpose()[['mean', 'std']]
        summary2 = turnover_df.apply(lambda x: x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])).transpose()
        pd.concat([summary1, summary2], axis=1).round(3).transpose().to_latex(f'turnover_{freq}.tex')

        appendix = '_after_fee'
    else:
        appendix = ''

    rs_df = pd.concat(rs + [SPY_freq], axis=1).apply(lambda x: (x + 1).cumprod())
    sns.set_theme(style="white")
    plt.clf()
    sns.lineplot(data=rs_df)
    sns.despine()
    plt.legend(loc='upper left', labels=['Dollar volume weighted',
                                         'Open interest weighted',
                                         'SPY'],
               frameon=False)
    # plt.title(f'Rebalance {freq.capitalize()}', fontdict={'fontsize': 25})
    x_max = rs_df.index.max()
    plt.xlabel("Year")
    plt.ylabel("Value in $")
    plt.tight_layout()
    plt.savefig(f'perf_{freq}{appendix}.png', dpi=300)

pd.concat(extremes, axis=1) \
    .apply(
    lambda x: x * 100 if x.name in ['r', 'port_r_ks|opt_vold_spread2', 'port_r_ks|open_interest_adj_spread2'] else x) \
    .round(2).to_latex(f'extremes{appendix}.tex')
# stats
stats = pd.DataFrame(perfs)[['sig', 'freq', 'Annualized Geometric Return', 'Annual Volatility',
                             'Annual Downside Deviation', 'Max Drawdown',
                             'Annual Sharpe Ratio', 'Annual Sortino Ratio', 'Annual Calmer Ratio']]
stats[['Max Drawdown', 'Annualized Geometric Return', 'Annual Downside Deviation', 'Annual Volatility']] = \
    stats[['Max Drawdown', 'Annualized Geometric Return', 'Annual Downside Deviation', 'Annual Volatility']] * 100
stats.sort_values('freq', ascending=False).round(2).transpose().to_latex(f'perfs{appendix}.tex')
