# Automatic Multiple Factor Evaluation
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame
from sig_evaluator import PortforlioAnalysis
from typing import Tuple, List


class AMFE:
    """
    Automatic multiple factor evaluation
    """

    def __init__(self, data: SparkDataFrame, name: str, time: str, sig_cols: List[str], r: str):
        self.data, self.name, self.time, self.sig_cols, self.r = data, name, time, sig_cols, r

    def add_forward_r(self, ends=[1, 5, 10, 20, 40],  start=1):
        for end in ends:
            col_name = 'forward_r_' + str(end)
            self.data = self.data \
                .withColumn(col_name,
                            f.sum(f.log(f.col(self.r) + 1)).over(
                                Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end))) \
                .withColumn(col_name, f.exp(f.col(col_name)) - 1)
        return self.data

    def add_sig_variations(self, diff_lag=1, avg_lookbacks=(5, 10, 20, 40)):
        for sig_col in self.sig_cols:
            # generate change in signals.
            self.data = self.data.withColumn('value_lagged', f.lag(f.col(sig_col), offset=diff_lag)
                                             .over(Window.partitionBy(self.name).orderBy(self.time))) \
                .withColumn(f'diff_{sig_col}', f.col(sig_col) - f.col('value_lagged')) \
                .drop('value_lagged')
            # generate average signals
            for lookback in avg_lookbacks:
                self.data = self.data.withColumn(f'avg_{sig_col}_{lookback}', f.mean(f.col(sig_col))
                                                 .over(Window.partitionBy(self.name).orderBy(self.time).rowsBetween(-lookback, 0)))
        return self.data

    def analyze(self, ntile=5, weight='mktcap'):
        perfs = []
        for j in [col for col in self.data.columns if 'forward_r_' in col]:
            for i in [col for col in self.data.columns if any(sig in col for sig in self.sig_cols)]:
                sigeva = PortforlioAnalysis(self.data, self.time, sig=i, r=j)
                univariate_r = sigeva.univariate_portfolio_r(ntile, weight)
                r=univariate_r.toPandas().sort_values(self.time)\
                    .groupby(['sig_rank', 'weight_scheme']).apply(lambda x: x[::10]).reset_index(drop=True)
                single_sort = sigeva.sort_return(r)
                single_sort2 = pd.concat([single_sort], keys=[j], names=['cum_r'])
                single_sort3 = pd.concat([single_sort2], keys=[i], names=['sig'])
                perfs.append(single_sort3)
        return pd.concat(perfs)
