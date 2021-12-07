import pyspark.sql.functions as f
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List


class OptionSignal:
    def __init__(self, data: SparkDataFrame, time: str, sec_name: str,cp_flag: str, call_str: str, put_str: str,
                 weight='None'):
        """

        Parameters
        ----------
        data : Spark dataframe
        time : column name for time
        sec_name : column name for stock names
        weight : column name for weight. If weight = 'None', then signals will be equally weighted. weight should be
         always positive.
        cp_flag : column name for call/put flag
        call_str : str value for call options.
        put_str : str value for put options.
        """

        self.data = data
        self.time = time
        self.sec_name = sec_name
        self.cp_flag = cp_flag
        self.call_str = call_str
        self.put_str = put_str
        if weight == 'None' or weight is None:
            self.data = self.data.withColumn('weight', f.lit(1))
            self.weight = 'weight'
        else:
            self.weight = weight

    def gen_aggregate_sig(self, sig: str, sig_agg: str = 'sig_agg',
                          group_extra_var_list: List[str] = None) -> SparkDataFrame:
        """
        Aggregate option level signals to stock level signal. Signals for both calls and puts need to be positive to
        correctly calculate average and spread.

        Parameters
        ----------
        sig : column name of the signal
        sig_agg : new column name of the aggregated signal
        group_extra_var_list : additional groupby variables other than self.time and self.sec_name

        Returns
        -------
        a Spark dataframe with groupby variables and five aggregated signals: a signal aggregated for all calls,
        a signal aggregated for all puts,  average signal aggregated for both calls and puts, a spread signal
        as the diffrence between aggregated calls and puts, a spread signal as the weighted average of both calls
        and puts with put signals as negative values.
        """

        if group_extra_var_list is None:
            group_vars = [self.time, self.sec_name]
        else:
            group_vars = [self.time, self.sec_name] + group_extra_var_list
        # average for calls, puts, or calls and puts
        sigs = {}
        for i in [self.call_str, self.put_str, 'avg']:
            data = self.data.filter(f.col(self.cp_flag) == i) if i in [self.call_str, self.put_str] else self.data
            sigs[i] = data.groupby(group_vars) \
                .agg(f.sum(f.col(sig) * f.abs(self.weight)).alias('numerator'),
                     f.sum(f.abs(self.weight)).alias('denominator')) \
                .withColumn(sig_agg + '_' + i, f.col('numerator') / f.col('denominator')) \
                .drop('numerator', 'denominator')
        # spread based on both calls and puts with put signals as negative
        data = self.data.withColumn(sig, f.when(f.col(self.cp_flag) == self.put_str, -1*f.col(sig)).otherwise(f.col(sig)))
        sigs['spread2'] = data.groupby(group_vars) \
            .agg(f.sum(f.col(sig) * f.abs(self.weight)).alias('numerator'),
                 f.sum(f.abs(self.weight)).alias('denominator')) \
            .withColumn(sig_agg + '_spread2', f.col('numerator') / f.col('denominator')) \
            .drop('numerator', 'denominator')

        return sigs[self.call_str] \
            .join(sigs[self.put_str], on=group_vars) \
            .join(sigs['avg'], on=group_vars) \
            .withColumn(sig_agg + '_ratio', f.col(sig_agg + '_' + self.call_str) / f.col(sig_agg + '_' + self.put_str)) \
            .withColumn(sig_agg + '_spread1', f.col(sig_agg + '_' + self.call_str) - f.col(sig_agg + '_' + self.put_str)) \
            .join(sigs['spread2'], on=group_vars)

     # option price and stock price ratio
    def gen_os_p(self, os_p, sig_name='os_p') -> SparkDataFrame:
        """
        os_p: option to stock price ratio
        """
        return self.gen_aggregate_sig(os_p, sig_name)

    # volume measures
    def gen_os_v(self, os_v: str, sig_name='os_v') -> SparkDataFrame:
        """
        os_v : column name for the ratio of option to stock volume
        """
        return self.gen_aggregate_sig(os_v, sig_name)

    def gen_vol(self, vol: str, sig_name='signed_vol') -> SparkDataFrame:
        """
        vol : column name for volume. Volume can be cash volume, put_call ratio, open interest, open interest change
        """
        return self.gen_aggregate_sig(vol, sig_name)

    # leverage measures
    def gen_ks(self, ks: str, sig_name='ks') -> SparkDataFrame:
        """
        ks : column name for the strike/stock price ratio
        """
        return self.gen_aggregate_sig(ks, sig_name)

    def gen_expdays(self, exp_days: str, sig_name='exp_days') -> SparkDataFrame:
        """
        exp_days : column name for expiration days
        """
        return self.gen_aggregate_sig(exp_days, sig_name)

    def gen_elasticity(self, elasticity: str, sig_name='elasticity') -> SparkDataFrame:
        """
        elasticity : column name for option elasticity
        """
        return self.gen_aggregate_sig(elasticity, sig_name)

    # IV measures
    def gen_IV(self, IV: str, sig_name='IV') -> SparkDataFrame:
        """
        IV: column name for implied volatility
        """
        return self.gen_aggregate_sig(IV, sig_name)

    def gen_IV_skew(self, IV: str, ks: str, sig_name='IV_skew') -> SparkDataFrame:
        """
        ks_larger_than_1_colname 's row values need to be True or False
        """
        # create a column to mark whether strike price is larger than stock price
        self.data = self.data \
            .withColumn('ks<=1', f.when(f.col(ks) <= 1, 'ks<=1').otherwise('ks>1'))
        # pivot by ks <= 1
        return self.gen_aggregate_sig(IV, 'IV', ['ks<=1'])\
            .drop('IV_spread') \
            .groupby(self.time, self.sec_name).pivot('ks<=1', ['ks<=1', 'ks>1']) \
            .agg(f.max('IV_C').alias('IV_C'), f.max('IV_P').alias('IV_P'), f.max('IV_avg').alias('IV_avg')) \
            .withColumn(sig_name + '_C_minus_C', f.col('ks<=1_IV_C') - f.col('ks>1_IV_C')) \
            .withColumn(sig_name + '_C_minus_P', f.col('ks<=1_IV_C') - f.col('ks>1_IV_P')) \
            .withColumn(sig_name + '_P_minus_P', f.col('ks<=1_IV_P') - f.col('ks>1_IV_P')) \
            .withColumn(sig_name + '_P_minus_C', f.col('ks<=1_IV_P') - f.col('ks>1_IV_C')) \
            .withColumn(sig_name + '_avg_minus_avg', f.col('ks<=1_IV_avg') - f.col('ks>1_IV_avg')) \
            .drop('ks<=1_IV_C', 'ks<=1_IV_P', 'ks<=1_IV_avg', 'ks>1_IV_C', 'ks>1_IV_P', 'ks>1_IV_avg')

    # VRP
    def gen_rv_iv_spread(self, rv_iv: str, sig_name='rv_iv_spread') -> SparkDataFrame:
        """
        rv_iv: column name for realized volatility minus implied volatility
        """
        return self.gen_aggregate_sig(rv_iv, sig_name)
