

class EDA:
    def __init__(self, df):
        self.df = df

    def describe(self, long=False):
        summary = self.df.describe()
        summary.loc['skew', :] = self.df.skew().tolist()
        summary.loc['kurtosis', :] = self.df.kurtosis().tolist()
        summary.loc['median', :] = self.df.median().tolist()
        summary.loc['dtypes', :] = self.df.dtypes.tolist()
        summary.loc['unique_values', :] = self.df.nunique().tolist()
        summary = summary.transpose()[['count', 'unique_values', 'dtypes', 'mean',  'median', 'std', 'skew', 'kurtosis','min',  'max']]
        if long:
            return summary.transpose()
        else:
            return summary






if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame({'a':[1, 34, 5, 6, 7, 7], 'b':[1, 3, 5, 6, 7, 7]})
    eda = EDA(df).describe()


