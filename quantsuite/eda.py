import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_datetime64_any_dtype
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df):
        self.df = df

    def describe_numerical(self, long=False):
        summary = self.df.describe()
        summary.loc['skew', :] = self.df.skew().tolist()
        summary.loc['kurtosis', :] = self.df.kurtosis().tolist()
        summary.loc['median', :] = self.df.median().tolist()
        summary = summary.transpose()[
            ['count', 'mean', 'median', 'std', 'skew', 'kurtosis', 'min', 'max']]
        if long:
            return summary.transpose()
        else:
            return summary

    def describe_all(self, long=False):
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
        fig.suptitle(title)
        for i in range(len(self.df.columns)):
            ax = axs[int(i / 2), i % 2]
            col = list(self.df.columns)[i]
            if is_numeric_dtype(self.df[col]) or is_datetime64_any_dtype(self.df[col]):
                ax.hist(self.df[col], bins=10)
            else:
                temp = self.df[col].value_counts().to_frame().reset_index()
                ax.bar(x=temp['index'], height=temp[col])
            ax.set_xlabel(col)
        return fig

if __name__ == '__main__':
    import pandas as pd

    df = pd.DataFrame({'a': [1, 34, 5, 6, 7, 7], 'b': [1, 3, 5, 6, 7, 7]})
    eda = EDA(df).describe()
