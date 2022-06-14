import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
import matplotlib.pyplot as plt
from typing import List

class EDA:
    def __init__(self, df):
        self.df = df

    def describe(self, num_cols:List[str], long=False):
        data = self.df[num_cols]
        summary = data.describe()
        summary.loc['skew', :] = data.skew(numeric_only=True).tolist()
        summary.loc['kurtosis', :] = data.kurtosis(numeric_only=True).tolist()
        summary.loc['median', :] = data.median(numeric_only=True).tolist()
        summary = summary.transpose()[
            ['count', 'mean', 'median', 'std', 'skew', 'kurtosis', 'min', 'max']]
        if long:
            return summary.transpose()
        else:
            return summary

    def info(self, long=False):
        res = pd.DataFrame({
                            'dtypes': self.df.dtypes.tolist(),
                            'unique_values': self.df.nunique().tolist(),
                            'pct_zero': (self.df == 0).sum(axis=0)/len(self.df)*100,
                            'pct_na': self.df.isna().sum(axis=0) / len(self.df) * 100
                            }, index = self.df.columns)
        if long:
            return res
        else:
            return res.transpose()

    def plot_hist(self, title = 'Histograms'):
        n_rows, n_cols = (len(self.df.columns) + 1) // 2, 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
        # make sure axs is a 2-d array even n_rows =  1
        axs = axs.reshape(n_rows, -1)
        fig.suptitle(title)
        for i in range(len(self.df.columns)):
            ax = axs[int(i / 2), i % 2]
            col = list(self.df.columns)[i]
            if (is_numeric_dtype(self.df[col]) or is_datetime64_any_dtype(self.df[col])) and \
                    not is_bool_dtype(self.df[col]):
                ax.hist(self.df[col], bins=10)
            else:
                temp = self.df[col].value_counts().to_frame().reset_index()
                ax.bar(x=temp['index'], height=temp[col])
            ax.set_xlabel(col)
        return fig

    def plot_boxplot(self,  num_cols:List[str], title = 'Boxplots'):
        n_rows, n_cols = (len(num_cols) + 1) // 2, 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
        # make sure axs is a 2-d array even n_rows =  1
        axs = axs.reshape(n_rows, -1)
        fig.suptitle(title)
        for i in range(len(num_cols)):
            ax = axs[int(i / 2), i % 2]
            col =  num_cols[i]
            ax.boxplot(self.df[col])
            ax.set_xlabel(col)
        return fig

if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('quantsuite/quantsuite/analytical_take_home_data_v3-1.csv')
    data['overspend'] = data['campaign_spend'] - data['campaign_budget']
    data['overspend_pct'] = data['overspend'] / data['campaign_budget']
    eda = EDA(data)
    eda.describe_numerical(['campaign_spend', 'campaign_budget', 'overspend', 'overspend_pct'])
    eda.plot_boxplot(['campaign_spend', 'campaign_budget', 'overspend', 'overspend_pct'])

