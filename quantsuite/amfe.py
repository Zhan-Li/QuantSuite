# Automatic Multiple Factor Evaluation
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame
from invtools.PerformanceEvaluation import PerformanceEvaluation
from typing import Tuple, List
from pandas import DataFrame as PandasDataFrame
import quantstats
from jinja2 import Environment, FileSystemLoader
from typing import Dict

class AMFE:
    """
    Automatic multiple factor evaluation
    """

    def __init__(self, data: SparkDataFrame, name: str, time: str, sig_cols: List[str], r: str, trade_fee: float):
        self.data, self.name, self.time, self.sig_cols, self.r = data, name, time, sig_cols, r
        self.trade_fee = trade_fee
        self.sorted_port_r = None
        self.port_r = None

    def add_forward_r(self, start:int, ends=List[int]):
        for end in ends:
            col_name = 'forward_r_' + str(end)
            window_gen_r = Window.partitionBy(self.name).orderBy(self.time).rowsBetween(start, end)
            self.data = self.data \
                .withColumn(col_name, f.sum(f.log(f.col(self.r) + 1)).over(window_gen_r)) \
                .withColumn(col_name, f.exp(f.col(col_name)) - 1)
        return self.data

    def add_sig_variations(self,  lookbacks=List[int], diff_lag=1, check_point = False):
        for sig_col in self.sig_cols:
            # generate change in signals.
            self.data = self.data\
                .withColumn('value_lagged', f.lag(f.col(sig_col), offset=diff_lag)
                                             .over(Window.partitionBy(self.name).orderBy(self.time))) \
                .withColumn(f'diff_{sig_col}', f.col(sig_col) - f.col('value_lagged')) \
                .drop('value_lagged')
            # generate average signals and g_score
            for lookback in lookbacks:
                self.data = self.data\
                    .withColumn(f'avg_{sig_col}_{lookback}', f.mean(sig_col)
                                                 .over(Window.partitionBy(self.name).orderBy(self.time).rowsBetween(-lookback, 0)))\
                    .withColumn(f'crossover_{sig_col}_{lookback}', f.col(sig_col)/f.col(f'avg_{sig_col}_{lookback}'))
            self.data = self.data.checkpoint() if check_point is True else self.data
        return self.data

    def analyze(self,  ntile:int, annual_SR_mult:int, vw = False, weight = None):
        perfs = []
        returns = []
        data = self.data.toPandas()
        for j in [col for col in data.columns if 'forward_r_' in col]:
            for i in [col for col in data.columns if any(sig in col for sig in self.sig_cols)]:
                sigeva = PerformanceEvaluation(data, self.time, self.name, sig=i, r=j,
                                            vw = vw, weight = weight)
                r = sigeva.univariate_portfolio_r(ntile, self.trade_fee)

                r2 = pd.concat([r], keys=[i], names=['sig'])
                r3 = pd.concat([r2], keys=[j], names=['forward_r'])
                returns.append(r3)

                single_sort = sigeva.sort_return(r, annual_SR_mult)
                single_sort2 = pd.concat([single_sort], keys=[i], names=['sig'])
                single_sort3 = pd.concat([single_sort2], keys=[j], names=['forward_r'])
                perfs.append(single_sort3)
        self.port_r=pd.concat(returns)
        self.sorted_port_r = pd.concat(perfs)

    def filter_sorted(self):
        results = self.sorted_port_r.xs('high_minus_low', level='sig_rank')
        return results\
            .loc[results['mean_return'].abs() - results['trade_cost'].abs() > 0]\
            .loc[results['pvalue'].abs() <= 0.05]

    def report(self, sig_params: Dict, html_template='reports/template.html', output='reports/report.html'):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(html_template)
        template_vars = {'signal_info': sig_params,
                         "sorted_r": self.sorted_port_r.to_html(),
                         "sorted_by_return": self.filter_sorted().sort_values('mean_return').to_html()}
        with open(output, "w") as f:
            f.write(template.render(template_vars))

    def tearsheet(self, r:str, nth:int, sig: str, weight:str, output):
        """
        nth: select every nth return
        """
        r_selected = self.port_r.loc[(r, sig)]
        r_final = r_selected.loc[(r_selected['sig_rank'] == 'high_minus_low')&(r_selected['weight_scheme'] == weight)] \
                        .drop_duplicates()\
                      .sort_values(self.time) \
                      .set_index(self.time)[r][::nth]
        quantstats.reports.html(r_final, output=output)