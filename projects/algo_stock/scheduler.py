import pandas as pd
import xmltodict
from yahoo_earnings_calendar import YahooEarningsCalendar
from datetime import date, timedelta
import importlib
import functions
importlib.reload(functions)
from functions import get_target_position
import numpy as np
from ib_insync import *
import json
import schedule
import time
import stock_hist
importlib.reload(stock_hist)
from stock_hist import download_stock_hist
import datetime

schedule.every().day.at("12:30").do(download_stock_hist)
print(datetime.datetime.now())

while True:
    schedule.run_pending()
    time.sleep(1)