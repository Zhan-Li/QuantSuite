import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame

class TechnicalSignal:
    """
    The OptionSignal class gnerates option-based trading signals.
    """
    def __init__(self, data: SparkDataFrame, time: str,  name: str):
        self.data, self.name, self.time = data, name, time

    def filter(self, df, start, end):
        """
        In rolling calculation, Pyspark will generate a value even the total number of obsevations is smaller than the
        window size in the beginning of the period.
        """
        return df\
            .withColumn('count', f.count(self.time).over(Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end)))\
            .filter(f.col('count') >= end-start+1)\
            .drop('count')

    def window(self, start, end):
        return Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end)

    def add_cumr(self, r, start, end, sig_name:str):
        """
        start, end as the star, end in rowsBetween, both start and end are inclusive.
        result: bad performance
        """
        self.data = self.data \
            .withColumn(sig_name, f.sum(f.log(f.col(r) + 1)).over(self.window(start, end))) \
            .withColumn(sig_name, f.exp(f.col(sig_name)) - 1)
        return self

    def add_comovement(self, r,  market_r:str, start, end, comovement:str):
        self.data = self.data\
            .withColumn('same_sign', f.when(f.signum(r) == f.signum(market_r), 1).otherwise(0))\
            .withColumn(comovement, f.mean(f.col('same_sign')).over(self.window(start, end)))\
            .drop('same_sign') \
            .withColumn('same_sign',
                        f.when((f.signum(r) == f.signum(market_r)) & (f.col(r) > 0) & (f.col(market_r) > 0), 1)
                        .otherwise(0))\
            .withColumn(comovement+'_up', f.mean(f.col('same_sign')).over(self.window(start, end))) \
            .drop('same_sign') \
            .withColumn('same_sign',
                        f.when((f.signum(r) == f.signum(market_r)) & (f.col(r) < 0) & (f.col(market_r) < 0),
                               1)
                        .otherwise(0)) \
            .withColumn(comovement+'_down', f.mean(f.col('same_sign')).over(self.window(start, end))) \
            .drop('same_sign')
        return self

    def add_correlation(self, r, market_r, start, end, corr):
        self.data = self.data\
            .withColumn(corr, f.corr(f.col(r), f.col(market_r)).over(self.window(start, end)))
        return self

    def add_bab(self):
        # betting against beta factor
        pass

    def add_illiq(self, r, vol_d, sig_name = 'illiq'):
        self.data = self.data\
            .withColumn(sig_name, f.abs(r)/f.col(vol_d))
        return self

    def add_max_min_r(self, r:str,  start:str, end:str, max_r:str, min_r:str):
        self.data = self.data\
            .withColumn(max_r, f.max(r).over(self.window(start, end)))\
            .withColumn(min_r, f.min(r).over(self.window(start, end)))
        return self

    def add_avg_r(self, r, start, end, past_avgr:str):
        self.data =  self.data\
            .withColumn(past_avgr, f.mean(r).over(self.window(start, end))) \
            .withColumn('indicator', f.when(f.col(r) > 0, 1).otherwise(0)) \
            .withColumn(past_avgr + '_up', f.mean(f.col(r) * f.col('indicator')).over(self.window(start, end))) \
            .drop('indicator') \
            .withColumn('indicator', f.when(f.col(r) < 0, 1).otherwise(0)) \
            .withColumn(past_avgr + '_down', f.mean(f.col(r) * f.col('indicator')).over(self.window(start, end))) \
            .drop('indicator')
        return self


    def add_std(self, value, start, end, std: str):
        self.data =  self.data\
            .withColumn(std, f.stddev(value).over(self.window(start, end)))\
            .withColumn('indicator', f.when(f.col(value) > 0, 1).otherwise(0))\
            .withColumn(std+'_up', f.stddev(f.col(value)*f.col('indicator')).over(self.window(start, end)))\
            .drop('indicator')\
            .withColumn('indicator', f.when(f.col(value) < 0, 1).otherwise(0)) \
            .withColumn(std + '_down', f.stddev(f.col(value) * f.col('indicator')).over(self.window(start, end))) \
            .drop('indicator')
        return self

    def add_value_average_ratio(self, value, start, end, ratio:str):
        """today'value divided by past average value
        Test result: Success!
        """
        self.data = self.data\
            .withColumn('avg', f.mean(value).over(self.window(start, end)))\
            .withColumn(ratio, f.col(value)/f.col('avg'))
        return self

    def add_drawdown(self, r:str,  drawdown = 'drawdown'):
        self.data =  self.data \
            .withColumn('cum_r', f.sum(f.log(f.col(r) + 1)).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn('cum_r', f.exp('cum_r'))\
            .withColumn('max_cum_r', f.max('cum_r').over(Window.partitionBy(self.name).orderBy(self.time)))\
            .withColumn(drawdown, f.col('cum_r')/f.col('max_cum_r') - 1)\
            .drop('cum_r', 'max_cum_r')
        return self

    def add_consecutive_r(self, r:str, n_r = 'n_r'):
        """
        Test result:  Long-short signal has a lower sharpe ratio than SPY during 2010 to 2020.
        """
        self.data = self.data\
            .withColumn('ret_sign', f.when(f.col(r) >= 0, 1).otherwise(0))\
            .withColumn('ret_sign_lag', f.lag('ret_sign').over(Window.partitionBy(self.name).orderBy(self.time)))\
            .withColumn('not_equal', f.when(f.col('ret_sign') != f.col('ret_sign_lag'), 1).otherwise(0)) \
            .withColumn('group', f.sum('not_equal').over(Window.partitionBy(self.name).orderBy(self.time)))\
            .withColumn(n_r, f.count(r).over(Window.partitionBy(self.name, 'group').orderBy(self.time)))\
            .withColumn(n_r, f.when(f.col(r) >= 0, f.col(n_r)).otherwise(-f.col(n_r)))
        return self

    def add_turnover(self, volume, num_shares, turnover:str):
        self.data = self.data\
            .withColumn(turnover, f.col(volume)/f.col(num_shares))
        return self

    def add_new_high(self, r):
        self.add_past_cumr(r,  Window.unboundedPreceding, Window.currentRow, 'cumr').data\
            .withColumn('max_cumr', f.max('cum_r').over(self.window(Window.unboundedPreceding, Window.currentRow)))\
            .withColumn('new_high')
        return self

    def add_drawdown_speed(self):
        pass

    def add_52w_high(self, prc_adj:str='prc_adj', sig_name = 'high_52w'):
        self.data = self.data \
            .withColumn(sig_name, f.max(prc_adj).over(self.window(-252, 0))) \
            .withColumn(sig_name, f.col(prc_adj)/f.col(sig_name))
        return self

    def add_spread(self, bid, ask, spread = 'spread'):
        self.data = self.data \
            .withColumn(spread, f.col(ask)/f.col(bid) - 1)
        return self




