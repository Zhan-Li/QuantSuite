import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame
from functools import reduce
from operator import add
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.io as pio
pio.templates.default = "ggplot2"


class PerformanceAnalytics:
    """
    The OptionSignal class gnerates option-based trading signals.
    """

    def __init__(self, data, time, r):
        self.data = data
        self.time = time
        self.r = r

    def plot_wealth(self, starting = 1, title = 'Cumulative Returns', output = 'ouput.html'):
        self.data['wealth'] = starting*(self.data[self.r] + 1).cumprod()
        px.line(self.data, x=self.time, y='wealth', title=title).write_html(output)







