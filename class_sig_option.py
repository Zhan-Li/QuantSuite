"""
For any signal involving stock data, you should first calculate the signal at the option level, and then aggregate.
"""
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List

class OptionSignal:
    """
    The OptionSignal class gnerates option-based trading signals.
    """
    def __init__(self, data, time, sec_name, cp_flag: str, call_str: str, put_str: str):
        """
        opt_file: option file
        sec_file: stock file
        time_col: time column, shared by both the opt_file and sec_file
        sec_id: stock indentifier, shared by both the opt_file and sec_file
        """
        self.data = data
        self.time = time
        self.sec_name = sec_name
        self.cp_flag = cp_flag
        self.call_str = call_str
        self.put_str = put_str

    def filter(self, signal_df, start, end):
        """
        In rolling calculation, Pyspark will generate a value even the total number of obsevations is smaller than the
        window size in the beginning of the period.
        """
        return signal_df\
            .withColumn('count', f.count(self.time).over(Window.partitionBy(self.sec_name).orderBy(self.time).rowsBetween(start, end)))\
            .filter(f.col('count') >= end-start+1)\
            .drop('count')

    def gen_aggregate_sig(self, sig:str, weight='None', sig_agg='sig_agg', group_extra_var_list=None):
        """
        sig: generally a positive number for calls and a negative number for puts
        weight: generally a positive number
        """
        if weight == 'None' or weight is None:
            self.data = self.data.withColumn('weight', f.lit(1))
            weight = 'weight'
        if group_extra_var_list is None:
            group_vars = [self.time, self.sec_name]
        else:
            group_vars = [self.time, self.sec_name] + group_extra_var_list

        sigs = {}
        for i in [self.call_str, self.put_str, 'CP']:
            data = self.data.filter(f.col(self.cp_flag) == i) if i in [self.call_str, self.put_str] else self.data
            sigs[i] = data.groupby(group_vars) \
                .agg(f.sum(f.col(sig) * f.abs(weight)).alias('numerator'),
                     f.sum(f.abs(weight)).alias('denominator')) \
                .withColumn(sig_agg + '_' + i, f.col('numerator') / f.col('denominator')) \
                .drop('numerator', 'denominator')
        return sigs[self.call_str]\
            .join(sigs[self.put_str], on=group_vars)\
            .join(sigs['CP'], on=group_vars)
    # volume measures
    def gen_os(self, os: str, weight: str, sig_name='os'):
        """
        os signal. Can be normalized by stock volume.
        weight: used to weight opt_vol, for example, elasticity
        result: put subsample has better result.
        initial optimal result: oi_adj_change as weight, CP or C as suffix, about 3.5 SR for daily data from 2015 to 2020
        """
        return self.gen_aggregate_sig(os, weight, sig_name)

    def gen_signed_vol(self, vol: str, weight: str, signed_vol='signed_vol'):
        """
        we can further have put-call volume ratio, time series ratios such as time series zscore or
         ratio to historical average
        """
        return self.gen_aggregate_sig(vol, weight, signed_vol)

    # leverage measures
    def gen_ks(self, ks:str, weight:str,  sig_name='ks'):
        return self.gen_aggregate_sig(ks, weight, sig_name)

    def gen_expdays(self, exp_days:str, weight:str, sig_name='exp_days'):
        return self.gen_aggregate_sig(exp_days, weight, sig_name)

    def gen_elasticity(self, elasticity:str, weight, sig_name='elasticity'):
        return self.gen_aggregate_sig(elasticity, weight, sig_name)

    # IV measures
    def gen_IV_spread(self, IV, weight, sig_name='IV_spread'):
        return self.gen_aggregate_sig(IV, weight, 'IV')\
            .withColumn(sig_name, f.col('IV'+ '_' + self.call_str) + f.col('IV' + '_' + self.put_str))\
            .drop('IV'+ '_' + self.call_str, 'IV' + '_' + self.put_str, 'IV_CP')

    def gen_IV_skew(self, IV: str, weight:str, ks_larger_than_1_colname: str, sig_name = 'IV_skew'):
        """
        ks_larger_than_1_colname 's row values need to be True or False
        """
        IVS = self.gen_aggregate_sig(IV, weight, 'IV', [ks_larger_than_1_colname])
        IVS_small_k = IVS.filter(f.col(ks_larger_than_1_colname) == False)\
            .withColumnRenamed('IV_'+ self.call_str, 'IV_' + self.call_str+'_small_k')\
            .withColumnRenamed('IV_' + self.put_str, 'IV_' + self.put_str + '_small_k')\
            .withColumnRenamed('IV_CP', 'IV_CP_small_k')\
            .drop(ks_larger_than_1_colname)
        IVS_large_k = IVS.filter(f.col(ks_larger_than_1_colname) == True) \
            .withColumnRenamed('IV_' + self.call_str, 'IV_' + self.call_str + '_large_k') \
            .withColumnRenamed('IV_' + self.put_str, 'IV_' + self.put_str + '_large_k') \
            .withColumnRenamed('IV_CP', 'IV_CP_large_k')\
            .drop(ks_larger_than_1_colname)
        return IVS_small_k.join(IVS_large_k, on=[self.time, self.sec_name], how='inner')\
            .withColumn(sig_name+'_C_minus_C', f.col('IV_'+self.call_str+'_small_k') - f.col('IV_' + self.call_str + '_large_k')) \
            .withColumn(sig_name+'_C_plus_P', f.col('IV_' + self.call_str + '_small_k') - f.col('IV_' + self.put_str + '_large_k')) \
            .withColumn(sig_name+'_P_minus_P', f.col('IV_' + self.put_str + '_small_k') - f.col('IV_' + self.put_str + '_large_k')) \
            .withColumn(sig_name+'_P_plus_C', f.col('IV_' + self.put_str + '_small_k') + f.col('IV_' + self.call_str + '_large_k')) \
            .withColumn(sig_name+'_CP_minus_CP', f.col('IV_CP_small_k') - f.col('IV_CP_large_k'))



    def gen_IV_TS(self):
        """
        IV term structure
        """
        pass

    def gen_IV_rank(self):
        pass

    # option_weighted_price to stock price ratio
    def gen_prc_ratio(self, opt_sig, opt_weight, cp_split, sec_data, sec_sig, ratio_name):
        self.data = self.add_aggregate_sig(opt_sig,  weight=opt_weight, cp_split=cp_split, sig_agg='opt_sig').data\
            .join(sec_data, on=[self.time, self.sec_name])\
            .withColumn(ratio_name, f.col('opt_sig')/f.col(sec_sig))
        return self
    #VRP
    def gen_rv_iv_spread(self, RV, IV, IV_weight, sec_data, sig_name = 'rv_iv_spread'):
        return self.gen_aggregate_sig(IV, IV_weight, 'IV') \
            .join(sec_data, on=[self.time, self.sec_name])\
            .withColumn(sig_name, f.col(RV) - f.col('IV_CP'))\
            .select(self.time, self.sec_name, sig_name)

