from pyspark.sql import  Window
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as f


class DataManipulator:
    """
    Manipulating pyspark dataframe
    """
    def __init__(self, data: SparkDataFrame, time: str, name: str):
        self.data, self.time, self.name = data, time, name

    def w(self, start:int, end:int):
        """
        wrapper for window, partiionby, orderby, rowsbetween
        """
        return Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end)

    def filter(self, df, start, end) -> SparkDataFrame:
        """
        In rolling calculation, Pyspark will generate a value even the total number of obsevations is smaller than the
        window size in the beginning of the period. This function filters these results with partial observation.
        """
        return df\
            .withColumn('count', f.count(self.time).over(self.w(start, end)))\
            .filter(f.col('count') >= end-start+1)\
            .drop('count')

    def add_sum(self, value: str, start: int, end: int, sum_col: str) -> 'DataManipulator':
        """
        end of period wealth assuming starting wealth of 1
        """
        self.data = self.data.withColumn(sum_col, f.sum(value).over(self.w(start, end)))
        return self

    def add_mean(self, value: str, start: int, end: int, avg_mean='avg') -> 'DataManipulator':
        self.data = self.data.withColumn(avg_mean, f.mean(value).over(self.w(start, end)))
        return self

    def add_change(self, value: str, lag: int=1, pct_change = False, change_col = 'change') -> 'DataManipulator':
        """
        This function adds two columns of changes and percentage changes.
        """
        self.data = self.data \
            .withColumn('value_lagged',
                        f.lag(f.col(value), offset=lag).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn(change_col, f.col(value) - f.col('value_lagged'))
        if pct_change is True:
            self.data = self.data.withColumn(change_col + '_pct', f.col(value)/f.col('value_lagged') - 1)
        self.data = self.data.drop('value_lagged')
        return self

    def add_acf(self, value: str, lag: int = 1, acf_col = 'acf'):
        """
        Autocorrelation
        """
        self.data = self.data.withColumn('value_lagged',
                        f.lag(f.col(value), offset=lag).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn(acf_col, f.corr(value, 'value_lagged'))\
            .drop('value_lagged')
        return self

    def add_std(self, value:str, start:int, end: int, std_col='std'):
        self.data = self.data.withColumn(std_col, f.stddev(value).over(self.w(start, end)))
        return self

    def add_skewness(self, value: str, start: int, end:int, skewness_col='skewness') -> 'DataManipulator':
        self.data = self.data.withColumn(skewness_col, f.skewness(value).over(self.w(start, end)))
        return self

    def add_kurtosis(self, value:str, start:int, end: int, kurtosis='kurtosis') -> 'DataManipulator':
        self.data  = self.data\
            .withColumn(kurtosis, f.kurtosis(value).over(self.w(start, end)))
        return self

    def add_rsd(self, value, start, end,  rsd_col = 'rsd')-> 'DataManipulator':
        """
        relative standard deviation
        """
        self.data = self.data\
            .withColumn('mean', f.mean(value).over(self.w(start, end))) \
            .withColumn('std', f.stddev(value).over(self.w(start, end)))\
            .withColumn(rsd_col, f.col('std')/f.col('mean'))\
            .drop('mean', 'std')
        return self

    def add_timeseries_comparison(self, value: str, start: int, end: int, zscore='zscore', ratio='ratio')-> 'DataManipulator':
        """
        generate time_series z-score and value to average ratio
        """
        self.data = self.data \
            .withColumn('avg', f.mean(value).over(self.w(start, end))) \
            .withColumn('std', f.stddev(value).over(self.w(start, end))) \
            .withColumn(zscore, (f.col(value) - f.col('avg')) / f.col('std')) \
            .withColumn(ratio, f.col(value)/f.col('avg'))\
            .drop('avg', 'std')
        return self


    def add_cross_section_zscore(self, value: str, zscore:str)-> 'DataManipulator':
        """
        generate time_series z-score
        return a df with a new column which is a zscore of another column
        """
        self.data = self.data \
            .withColumn('avg', f.mean(value).over(Window.partitionBy(self.time))) \
            .withColumn('std', f.stddev(value).over(Window.partitionBy(self.time))) \
            .withColumn(zscore, (f.col(value) - f.col('avg')) / f.col('std')) \
            .drop('avg', 'std')
        return self

    def resample(self, nth: int)-> 'DataManipulator':
        """
        nth: resample very nth row
        """
        self.data = self.data \
            .withColumn('index', f.row_number().over(Window.partitionBy(self.name).orderBy(self.time))) \
            .filter(f.col('index') % nth == 0)\
            .drop('index')
        return self

    def add_wealth(self, r: str, start: int, end: int, wealth: str)-> 'DataManipulator':
        """
        end of period wealth assuming starting wealth of 1
        """
        self.data = self.data \
            .withColumn(wealth, f.sum(f.log(f.col(r) + 1)).over(self.w(start, end))) \
            .withColumn(wealth, f.exp(f.col(wealth)))
        return self


    def prc_to_r(self, price: str, r: str, fee: float, lag=1)-> 'DataManipulator':
        """
        fee: actual fee, not percentage
        """
        self.data = self.data \
            .withColumn('value_lagged', f.lag(f.col(price), offset=lag).over(Window.partitionBy(self.name).orderBy(self.time))) \
            .withColumn(r, (f.col(price)*100 - f.col('value_lagged')*100 - 2*fee)/(f.col('value_lagged')*100 )) \
            .drop('value_lagged')
        return self



