import schedule
import time
from stock_hist import download_stock_hist
import datetime

schedule.every().day.at("12:30").do(download_stock_hist)
print(datetime.datetime.now())

while True:
    schedule.run_pending()
    time.sleep(1)