import datetime
import time

import schedule

import download_stock_data

schedule.every().day.at("12:30").do(download_stock_data)
print(datetime.datetime.now())

while True:
    schedule.run_pending()
    time.sleep(1)
